//! MCP transport implementations.
//!
//! This module provides transport abstractions for MCP communication.

use crate::error::{McpError, McpResult};
use crate::types::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex};

/// Trait for MCP transport implementations.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a request and wait for response.
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse>;

    /// Send a notification (no response expected).
    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()>;

    /// Close the transport.
    async fn close(&self) -> McpResult<()>;

    /// Check if the transport is connected.
    fn is_connected(&self) -> bool;
}

/// Stdio transport for local MCP servers.
///
/// This transport communicates with an MCP server via stdin/stdout.
pub struct StdioTransport {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<ChildStdin>>,
    response_rx: Arc<Mutex<mpsc::Receiver<String>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl StdioTransport {
    /// Spawn a new process and connect via stdio.
    pub async fn spawn(command: &str, args: &[&str]) -> McpResult<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| McpError::Transport(format!("Failed to spawn {}: {}", command, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("No stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("No stdout".to_string()))?;

        // Create channel for responses
        let (tx, rx) = mpsc::channel(100);

        // Spawn reader task
        tokio::spawn(Self::reader_task(stdout, tx));

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(stdin)),
            response_rx: Arc::new(Mutex::new(rx)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    async fn reader_task(stdout: ChildStdout, tx: mpsc::Sender<String>) {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        if tx.send(trimmed).await.is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }

    async fn send_raw(&self, data: &str) -> McpResult<()> {
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(data.as_bytes())
            .await
            .map_err(|e| McpError::Io(e))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::Io(e))?;
        stdin.flush().await.map_err(|e| McpError::Io(e))?;
        Ok(())
    }

    async fn receive_raw(&self) -> McpResult<String> {
        let mut rx = self.response_rx.lock().await;
        rx.recv()
            .await
            .ok_or_else(|| McpError::ConnectionClosed)
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        let json = serde_json::to_string(request)?;
        self.send_raw(&json).await?;

        let line = self.receive_raw().await?;
        let response: JsonRpcResponse = serde_json::from_str(&line)?;
        Ok(response)
    }

    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()> {
        let json = serde_json::to_string(notification)?;
        self.send_raw(&json).await
    }

    async fn close(&self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);

        let mut child = self.child.lock().await;
        if let Some(mut c) = child.take() {
            c.kill().await.ok();
        }
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// HTTP transport for remote MCP servers.
///
/// This transport communicates with an MCP server via HTTP.
#[cfg(feature = "reqwest")]
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    session_id: Arc<Mutex<Option<String>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "reqwest")]
impl HttpTransport {
    /// Create a new HTTP transport.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            session_id: Arc::new(Mutex::new(None)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Create with custom client.
    pub fn with_client(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        Self {
            client,
            base_url: base_url.into(),
            session_id: Arc::new(Mutex::new(None)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }
}

#[cfg(feature = "reqwest")]
#[async_trait]
impl McpTransport for HttpTransport {
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        let url = format!("{}/message", self.base_url);
        let mut req = self.client.post(&url).json(request);

        // Add session ID if we have one
        let session_id = self.session_id.lock().await;
        if let Some(ref id) = *session_id {
            req = req.header("X-Session-Id", id);
        }
        drop(session_id);

        let response = req.send().await.map_err(|e| McpError::Transport(e.to_string()))?;

        // Store session ID from response
        if let Some(id) = response.headers().get("X-Session-Id") {
            if let Ok(id_str) = id.to_str() {
                *self.session_id.lock().await = Some(id_str.to_string());
            }
        }

        if !response.status().is_success() {
            return Err(McpError::Http(response.status().as_u16()));
        }

        let json_response: JsonRpcResponse = response
            .json()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        Ok(json_response)
    }

    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()> {
        let url = format!("{}/message", self.base_url);
        let mut req = self.client.post(&url).json(notification);

        let session_id = self.session_id.lock().await;
        if let Some(ref id) = *session_id {
            req = req.header("X-Session-Id", id);
        }
        drop(session_id);

        let response = req.send().await.map_err(|e| McpError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(McpError::Http(response.status().as_u16()));
        }

        Ok(())
    }

    async fn close(&self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Memory transport for testing.
pub struct MemoryTransport {
    responses: Arc<Mutex<std::collections::VecDeque<JsonRpcResponse>>>,
    requests: Arc<Mutex<Vec<JsonRpcRequest>>>,
    notifications: Arc<Mutex<Vec<JsonRpcNotification>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for MemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTransport {
    /// Create a new memory transport.
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            requests: Arc::new(Mutex::new(Vec::new())),
            notifications: Arc::new(Mutex::new(Vec::new())),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Add a response to be returned.
    pub async fn push_response(&self, response: JsonRpcResponse) {
        self.responses.lock().await.push_back(response);
    }

    /// Get recorded requests.
    pub async fn get_requests(&self) -> Vec<JsonRpcRequest> {
        self.requests.lock().await.clone()
    }

    /// Get recorded notifications.
    pub async fn get_notifications(&self) -> Vec<JsonRpcNotification> {
        self.notifications.lock().await.clone()
    }

    /// Clear all recorded data.
    pub async fn clear(&self) {
        self.responses.lock().await.clear();
        self.requests.lock().await.clear();
        self.notifications.lock().await.clear();
    }
}

#[async_trait]
impl McpTransport for MemoryTransport {
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        self.requests.lock().await.push(request.clone());

        self.responses
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| McpError::NoResult)
    }

    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()> {
        self.notifications.lock().await.push(notification.clone());
        Ok(())
    }

    async fn close(&self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_transport() {
        let transport = MemoryTransport::new();

        // Push a response
        let response = JsonRpcResponse::success(1, "test");
        transport.push_response(response).await;

        // Make request
        let request = JsonRpcRequest::new(1, "test");
        let resp = transport.request(&request).await.unwrap();

        assert!(!resp.is_error());

        // Check recorded requests
        let requests = transport.get_requests().await;
        assert_eq!(requests.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_transport_notification() {
        let transport = MemoryTransport::new();

        let notification = JsonRpcNotification::new("test/notify");
        transport.notify(&notification).await.unwrap();

        let notifications = transport.get_notifications().await;
        assert_eq!(notifications.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_transport_close() {
        let transport = MemoryTransport::new();
        assert!(transport.is_connected());

        transport.close().await.unwrap();
        assert!(!transport.is_connected());
    }
}
