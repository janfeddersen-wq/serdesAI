//! Server-Sent Events (SSE) parsing.
//!
//! This module provides utilities for parsing SSE streams from HTTP responses.

use crate::error::{StreamError, StreamResult};
use crate::partial_response::ResponseDelta;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;
use serde::de::DeserializeOwned;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

const MAX_BUFFER_SIZE: usize = 10 * 1024 * 1024;

/// A parsed SSE event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// Event type (if specified).
    pub event: Option<String>,
    /// Event data.
    pub data: String,
    /// Event ID (if specified).
    pub id: Option<String>,
    /// Retry timeout (if specified).
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create a new SSE event with just data.
    pub fn data(data: impl Into<String>) -> Self {
        Self {
            event: None,
            data: data.into(),
            id: None,
            retry: None,
        }
    }

    /// Set the event type.
    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    /// Set the event ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Check if this is a "done" event (e.g., [DONE]).
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]" || self.event.as_deref() == Some("done")
    }

    /// Parse the data as JSON.
    pub fn parse_data<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.data)
    }
}

/// Parser for Server-Sent Events streams.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: String,
    events: VecDeque<SseEvent>,
    last_event_id: Option<String>,
}

impl SseParser {
    /// Create a new SSE parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes into the parser.
    pub fn feed(&mut self, bytes: &Bytes) -> StreamResult<Vec<SseEvent>> {
        let chunk = String::from_utf8_lossy(bytes);
        self.feed_str(&chunk)
    }

    /// Feed a string into the parser.
    pub fn feed_str(&mut self, s: &str) -> StreamResult<Vec<SseEvent>> {
        self.buffer.push_str(s);

        if self.buffer.len() > MAX_BUFFER_SIZE {
            return Err(StreamError::BufferOverflow);
        }

        self.parse_buffer()
    }

    /// Call when stream ends to flush any remaining event.
    pub fn finish(&mut self) -> StreamResult<Vec<SseEvent>> {
        let mut events = self.parse_buffer()?;

        if !self.buffer.trim().is_empty() {
            if let Some(event) = self.parse_event(self.buffer.trim_end_matches(['\n', '\r'])) {
                if let Some(id) = &event.id {
                    self.last_event_id = Some(id.clone());
                }
                self.events.push_back(event.clone());
                events.push(event);
            }
        }

        self.buffer.clear();

        Ok(events)
    }

    /// Get the next parsed event.
    pub fn next_event(&mut self) -> Option<SseEvent> {
        self.events.pop_front()
    }

    /// Check if there are pending events.
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Get the last event ID.
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// Clear the parser state.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.events.clear();
    }

    fn parse_buffer(&mut self) -> StreamResult<Vec<SseEvent>> {
        let mut parsed_events = Vec::new();

        // Split by double newlines (event boundaries)
        while let Some((pos, delimiter_len)) = self.find_event_boundary() {
            let event_str = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + delimiter_len..].to_string();
            self.buffer = self.buffer.trim_start_matches(['\n', '\r']).to_string();

            if let Some(event) = self.parse_event(&event_str) {
                if let Some(id) = &event.id {
                    self.last_event_id = Some(id.clone());
                }
                self.events.push_back(event.clone());
                parsed_events.push(event);
            }
        }

        Ok(parsed_events)
    }

    fn find_event_boundary(&self) -> Option<(usize, usize)> {
        let newline = self.buffer.find("\n\n").map(|pos| (pos, 2));
        let carriage = self.buffer.find("\r\n\r\n").map(|pos| (pos, 4));

        match (newline, carriage) {
            (Some(nl), Some(cr)) => Some(if cr.0 < nl.0 { cr } else { nl }),
            (Some(nl), None) => Some(nl),
            (None, Some(cr)) => Some(cr),
            (None, None) => None,
        }
    }

    fn parse_event(&self, s: &str) -> Option<SseEvent> {
        let mut event = None;
        let mut data_lines = Vec::new();
        let mut id = None;
        let mut retry = None;

        for line in s.lines() {
            if line.is_empty() || line.starts_with(':') {
                // Comment or empty line
                continue;
            }

            if let Some(value) = line.strip_prefix("event:") {
                event = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                data_lines.push(value.trim_start().to_string());
            } else if let Some(value) = line.strip_prefix("id:") {
                id = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("retry:") {
                retry = value.trim().parse().ok();
            } else if line.starts_with("data") {
                // "data" without colon means empty data line
                data_lines.push(String::new());
            }
        }

        if data_lines.is_empty() {
            return None;
        }

        Some(SseEvent {
            event,
            data: data_lines.join("\n"),
            id,
            retry,
        })
    }
}

pin_project! {
    /// Stream adapter that parses SSE from a byte stream.
    pub struct SseStream<S> {
        #[pin]
        inner: S,
        parser: SseParser,
        finished: bool,
    }
}

impl<S> SseStream<S>
where
    S: Stream<Item = Result<Bytes, std::io::Error>>,
{
    /// Create a new SSE stream from a byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            parser: SseParser::new(),
            finished: false,
        }
    }
}

impl<S> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, std::io::Error>> + Unpin,
{
    type Item = StreamResult<SseEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Return buffered events first
        if let Some(event) = this.parser.next_event() {
            return Poll::Ready(Some(Ok(event)));
        }

        if *this.finished {
            return Poll::Ready(None);
        }

        // Poll for more data
        match this.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                if let Err(error) = this.parser.feed(&bytes) {
                    return Poll::Ready(Some(Err(error)));
                }

                if let Some(event) = this.parser.next_event() {
                    Poll::Ready(Some(Ok(event)))
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(StreamError::Io(e)))),
            Poll::Ready(None) => {
                *this.finished = true;

                if let Err(error) = this.parser.finish() {
                    return Poll::Ready(Some(Err(error)));
                }

                // Return any remaining events
                if let Some(event) = this.parser.next_event() {
                    Poll::Ready(Some(Ok(event)))
                } else {
                    Poll::Ready(None)
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Extension trait for converting SSE events to response deltas.
pub trait SseEventExt {
    /// Convert to a response delta (if possible).
    fn to_response_delta(&self) -> Option<ResponseDelta>;
}

impl SseEventExt for SseEvent {
    fn to_response_delta(&self) -> Option<ResponseDelta> {
        if self.is_done() {
            return Some(ResponseDelta::Finish {
                reason: serdes_ai_core::FinishReason::Stop,
            });
        }

        // Try to parse as JSON and extract delta
        // This is provider-specific, so we attempt common formats
        if let Ok(json) = self.parse_data::<serde_json::Value>() {
            // OpenAI format: choices[0].delta
            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    if let Some(delta) = choice.get("delta") {
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            return Some(ResponseDelta::Text {
                                index: 0,
                                content: content.to_string(),
                            });
                        }
                    }
                }
            }

            // Anthropic format: type = "content_block_delta"
            if json.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                if let Some(delta) = json.get("delta") {
                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                        let index =
                            json.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                        return Some(ResponseDelta::Text {
                            index,
                            content: text.to_string(),
                        });
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser_basic() {
        let mut parser = SseParser::new();
        parser.feed_str("data: hello\n\n").unwrap();

        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "hello");
        assert!(event.event.is_none());
    }

    #[test]
    fn test_sse_parser_with_event_type() {
        let mut parser = SseParser::new();
        parser.feed_str("event: message\ndata: hello\n\n").unwrap();

        let event = parser.next_event().unwrap();
        assert_eq!(event.event, Some("message".to_string()));
        assert_eq!(event.data, "hello");
    }

    #[test]
    fn test_sse_parser_multiline_data() {
        let mut parser = SseParser::new();
        parser.feed_str("data: line1\ndata: line2\n\n").unwrap();

        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "line1\nline2");
    }

    #[test]
    fn test_sse_parser_multiple_events() {
        let mut parser = SseParser::new();
        parser.feed_str("data: first\n\ndata: second\n\n").unwrap();

        let event1 = parser.next_event().unwrap();
        let event2 = parser.next_event().unwrap();

        assert_eq!(event1.data, "first");
        assert_eq!(event2.data, "second");
        assert!(parser.next_event().is_none());
    }

    #[test]
    fn test_sse_parser_with_id() {
        let mut parser = SseParser::new();
        parser.feed_str("id: 123\ndata: hello\n\n").unwrap();

        let event = parser.next_event().unwrap();
        assert_eq!(event.id, Some("123".to_string()));
        assert_eq!(parser.last_event_id(), Some("123"));
    }

    #[test]
    fn test_sse_parser_with_retry() {
        let mut parser = SseParser::new();
        parser.feed_str("retry: 5000\ndata: hello\n\n").unwrap();

        let event = parser.next_event().unwrap();
        assert_eq!(event.retry, Some(5000));
    }

    #[test]
    fn test_sse_parser_ignores_comments() {
        let mut parser = SseParser::new();
        parser
            .feed_str(": this is a comment\ndata: hello\n\n")
            .unwrap();

        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "hello");
    }

    #[test]
    fn test_sse_event_is_done() {
        let event = SseEvent::data("[DONE]");
        assert!(event.is_done());

        let event = SseEvent::data("hello");
        assert!(!event.is_done());

        let event = SseEvent::data("something").with_event("done");
        assert!(event.is_done());
    }

    #[test]
    fn test_sse_event_parse_data() {
        let event = SseEvent::data("{\"key\": \"value\"}");
        let parsed: serde_json::Value = event.parse_data().unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_sse_parser_incremental() {
        let mut parser = SseParser::new();

        // Feed partial data
        parser.feed_str("data: hel").unwrap();
        assert!(parser.next_event().is_none());

        parser.feed_str("lo\n\n").unwrap();
        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "hello");
    }

    #[test]
    fn test_sse_to_response_delta() {
        // Test [DONE] event
        let done_event = SseEvent::data("[DONE]");
        let delta = done_event.to_response_delta().unwrap();
        assert!(matches!(delta, ResponseDelta::Finish { .. }));

        // Test OpenAI format
        let openai_event = SseEvent::data(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#);
        let delta = openai_event.to_response_delta().unwrap();
        if let ResponseDelta::Text { content, .. } = delta {
            assert_eq!(content, "Hello");
        } else {
            panic!("Expected text delta");
        }
    }
}
