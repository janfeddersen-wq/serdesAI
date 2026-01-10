//! WebSocket streaming support.
//!
//! This module provides WebSocket-based streaming for real-time communication
//! with LLM providers that support WebSocket connections.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_streaming::websocket::{WebSocketStream, WebSocketConfig};
//!
//! let config = WebSocketConfig::new("wss://api.example.com/v1/stream");
//! let mut stream = WebSocketStream::connect(config).await?;
//!
//! while let Some(message) = stream.next().await {
//!     match message? {
//!         WsMessage::Text(text) => println!("Received: {}", text),
//!         WsMessage::Close => break,
//!         _ => {}
//!     }
//! }
//! ```

use futures::{SinkExt, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async, tungstenite::protocol::Message as WsMessage, MaybeTlsStream,
    WebSocketStream as TungsteniteStream,
};

use crate::error::{StreamError, StreamResult};
use crate::events::AgentStreamEvent;
use crate::partial_response::ResponseDelta;

/// WebSocket connection configuration.
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// WebSocket URL.
    pub url: String,
    /// Optional headers for the connection.
    pub headers: Vec<(String, String)>,
    /// Ping interval in seconds.
    pub ping_interval: Option<u64>,
    /// Connection timeout in seconds.
    pub timeout: u64,
}

impl WebSocketConfig {
    /// Create a new WebSocket config with the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: Vec::new(),
            ping_interval: Some(30),
            timeout: 30,
        }
    }

    /// Add a header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    /// Set authorization header.
    pub fn with_auth(self, token: impl Into<String>) -> Self {
        self.with_header("Authorization", format!("Bearer {}", token.into()))
    }

    /// Set ping interval.
    pub fn with_ping_interval(mut self, seconds: u64) -> Self {
        self.ping_interval = Some(seconds);
        self
    }

    /// Disable ping.
    pub fn without_ping(mut self) -> Self {
        self.ping_interval = None;
        self
    }

    /// Set connection timeout.
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout = seconds;
        self
    }
}

/// WebSocket message types.
#[derive(Debug, Clone)]
pub enum WsStreamMessage {
    /// Text message containing JSON delta.
    Text(String),
    /// Binary message.
    Binary(Vec<u8>),
    /// Ping frame.
    Ping,
    /// Pong frame.
    Pong,
    /// Connection closed.
    Close,
}

impl From<WsMessage> for WsStreamMessage {
    fn from(msg: WsMessage) -> Self {
        match msg {
            WsMessage::Text(text) => WsStreamMessage::Text(text.to_string()),
            WsMessage::Binary(data) => WsStreamMessage::Binary(data.to_vec()),
            WsMessage::Ping(_) => WsStreamMessage::Ping,
            WsMessage::Pong(_) => WsStreamMessage::Pong,
            WsMessage::Close(_) => WsStreamMessage::Close,
            WsMessage::Frame(_) => WsStreamMessage::Binary(vec![]),
        }
    }
}

/// WebSocket stream wrapper for LLM streaming.
pub struct WebSocketStream {
    inner: TungsteniteStream<MaybeTlsStream<TcpStream>>,
    config: WebSocketConfig,
}

impl WebSocketStream {
    /// Connect to a WebSocket endpoint.
    pub async fn connect(config: WebSocketConfig) -> StreamResult<Self> {
        let (ws_stream, _) = connect_async(&config.url)
            .await
            .map_err(|e| StreamError::Connection(e.to_string()))?;

        Ok(Self {
            inner: ws_stream,
            config,
        })
    }

    /// Send a text message.
    pub async fn send_text(&mut self, text: impl Into<String>) -> StreamResult<()> {
        self.inner
            .send(WsMessage::Text(text.into()))
            .await
            .map_err(|e| StreamError::Send(e.to_string()))
    }

    /// Send a JSON message.
    pub async fn send_json<T: serde::Serialize>(&mut self, value: &T) -> StreamResult<()> {
        let json =
            serde_json::to_string(value).map_err(|e| StreamError::Serialization(e.to_string()))?;
        self.send_text(json).await
    }

    /// Close the connection.
    pub async fn close(&mut self) -> StreamResult<()> {
        self.inner
            .close(None)
            .await
            .map_err(|e| StreamError::Connection(e.to_string()))
    }

    /// Receive the next message.
    pub async fn next_message(&mut self) -> Option<StreamResult<WsStreamMessage>> {
        match self.inner.next().await {
            Some(Ok(msg)) => Some(Ok(msg.into())),
            Some(Err(e)) => Some(Err(StreamError::Receive(e.to_string()))),
            None => None,
        }
    }

    /// Try to parse text messages as response deltas.
    pub async fn next_delta(&mut self) -> Option<StreamResult<ResponseDelta>> {
        loop {
            match self.next_message().await? {
                Ok(WsStreamMessage::Text(text)) => {
                    match serde_json::from_str::<ResponseDelta>(&text) {
                        Ok(delta) => return Some(Ok(delta)),
                        Err(e) => {
                            tracing::warn!("Failed to parse WebSocket message as delta: {}", e);
                            continue;
                        }
                    }
                }
                Ok(WsStreamMessage::Close) => return None,
                Ok(WsStreamMessage::Ping) | Ok(WsStreamMessage::Pong) => continue,
                Ok(_) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &WebSocketConfig {
        &self.config
    }
}

impl futures::Stream for WebSocketStream {
    type Item = StreamResult<WsStreamMessage>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(msg))) => Poll::Ready(Some(Ok(msg.into()))),
            Poll::Ready(Some(Err(e))) => {
                Poll::Ready(Some(Err(StreamError::Receive(e.to_string()))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Convert WebSocket stream into an agent stream.
pub struct WebSocketAgentStream {
    ws: WebSocketStream,
    run_id: String,
}

impl WebSocketAgentStream {
    /// Create a new agent stream from a WebSocket connection.
    pub fn new(ws: WebSocketStream, run_id: impl Into<String>) -> Self {
        Self {
            ws,
            run_id: run_id.into(),
        }
    }

    /// Receive the next agent stream event.
    pub async fn next_event(&mut self) -> Option<StreamResult<AgentStreamEvent>> {
        loop {
            match self.ws.next_delta().await? {
                Ok(delta) => {
                    // Convert delta to agent stream event
                    match delta {
                        ResponseDelta::Text { index, content } => {
                            return Some(Ok(AgentStreamEvent::TextDelta {
                                content,
                                part_index: index,
                            }));
                        }
                        ResponseDelta::ToolCall {
                            index,
                            name,
                            args,
                            id,
                        } => {
                            if let Some(name) = name {
                                return Some(Ok(AgentStreamEvent::ToolCallStart {
                                    name,
                                    tool_call_id: id,
                                    index,
                                }));
                            } else if let Some(args) = args {
                                return Some(Ok(AgentStreamEvent::ToolCallDelta {
                                    args_delta: args,
                                    index,
                                }));
                            } else {
                                // Skip empty tool call deltas, continue loop
                                continue;
                            }
                        }
                        ResponseDelta::Thinking { index, content, .. } => {
                            return Some(Ok(AgentStreamEvent::ThinkingDelta { content, index }));
                        }
                        ResponseDelta::Finish { .. } => {
                            return Some(Ok(AgentStreamEvent::RunComplete {
                                run_id: self.run_id.clone(),
                                total_steps: 1,
                            }));
                        }
                        ResponseDelta::Usage { usage } => {
                            return Some(Ok(AgentStreamEvent::UsageUpdate { usage }));
                        }
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = WebSocketConfig::new("wss://example.com/stream")
            .with_auth("token123")
            .with_timeout(60)
            .with_ping_interval(15);

        assert_eq!(config.url, "wss://example.com/stream");
        assert_eq!(config.timeout, 60);
        assert_eq!(config.ping_interval, Some(15));
        assert!(config.headers.iter().any(|(k, _)| k == "Authorization"));
    }

    #[test]
    fn test_ws_message_conversion() {
        let text_msg = WsMessage::Text("hello".to_string());
        let converted: WsStreamMessage = text_msg.into();
        assert!(matches!(converted, WsStreamMessage::Text(s) if s == "hello"));

        let close_msg = WsMessage::Close(None);
        let converted: WsStreamMessage = close_msg.into();
        assert!(matches!(converted, WsStreamMessage::Close));
    }
}
