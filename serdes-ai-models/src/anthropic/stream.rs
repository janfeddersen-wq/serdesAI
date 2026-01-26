//! Anthropic SSE stream parser.
//!
//! This module provides streaming support for Anthropic's Messages API.

use super::types::{ContentBlockDelta, ContentBlockStart, StreamEvent};
use crate::error::ModelError;
use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;
use serdes_ai_core::messages::{
    ModelResponsePartDelta, ModelResponseStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent,
    TextPart, ThinkingPart, ThinkingPartDelta, ToolCallPart,
};
use serdes_ai_core::ModelResponsePart;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    /// Anthropic SSE stream parser.
    pub struct AnthropicStreamParser<S> {
        #[pin]
        inner: S,
        buffer: String,
        // Track content blocks in progress
        blocks: HashMap<usize, BlockState>,
        // Message metadata
        message_id: Option<String>,
        model: Option<String>,
        // Usage tracking
        input_tokens: u64,
        output_tokens: u64,
        cache_creation_tokens: Option<u64>,
        cache_read_tokens: Option<u64>,
        // Finished
        done: bool,
    }
}

/// State for an in-progress content block.
#[derive(Debug, Clone)]
enum BlockState {
    Text {
        content: String,
    },
    ToolUse {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        name: String,
        input_json: String,
    },
    Thinking {
        content: String,
        signature: Option<String>,
    },
    RedactedThinking {
        /// The encrypted signature data.
        #[allow(dead_code)]
        signature: String,
    },
}

impl<S> AnthropicStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    /// Create a new stream parser.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            blocks: HashMap::new(),
            message_id: None,
            model: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            done: false,
        }
    }
}

impl<S> Stream for AnthropicStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.done {
            return Poll::Ready(None);
        }

        loop {
            // Check if we have complete events in the buffer
            // Anthropic uses "event: type\ndata: json\n\n" format
            while let Some(event_result) = parse_next_event(this.buffer) {
                match event_result {
                    Ok((event_type, data)) => {
                        if let Some(result) = process_event(
                            &event_type,
                            &data,
                            this.blocks,
                            this.message_id,
                            this.model,
                            this.input_tokens,
                            this.output_tokens,
                            this.cache_creation_tokens,
                            this.cache_read_tokens,
                            this.done,
                        ) {
                            return Poll::Ready(Some(result));
                        }
                    }
                    Err(e) => {
                        return Poll::Ready(Some(Err(e)));
                    }
                }
            }

            // Need more data
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        this.buffer.push_str(text);
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ModelError::Other(e.into()))));
                }
                Poll::Ready(None) => {
                    *this.done = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Parse the next complete event from the buffer.
/// Returns Some((event_type, data)) if a complete event was found.
fn parse_next_event(buffer: &mut String) -> Option<Result<(String, String), ModelError>> {
    // Look for event: and data: lines followed by blank line
    let mut event_type = None;
    let mut data = None;
    let mut end_pos = 0;
    let mut found_event = false;

    for (i, line) in buffer.lines().enumerate() {
        if let Some(stripped) = line.strip_prefix("event: ") {
            event_type = Some(stripped.to_string());
        } else if let Some(stripped) = line.strip_prefix("data: ") {
            data = Some(stripped.to_string());
        } else if line.is_empty() && (event_type.is_some() || data.is_some()) {
            // Calculate position after this empty line
            end_pos = buffer
                .lines()
                .take(i + 1)
                .map(|l| l.len() + 1) // +1 for newline
                .sum();
            found_event = true;
            break;
        }
    }

    if found_event {
        // Remove processed content from buffer
        buffer.drain(..end_pos.min(buffer.len()));

        match (event_type, data) {
            (Some(et), Some(d)) => return Some(Ok((et, d))),
            (None, Some(d)) => return Some(Ok(("message".to_string(), d))),
            _ => {}
        }
    }

    None
}

/// Process a parsed event into a stream event.
#[allow(clippy::too_many_arguments)]
fn process_event(
    _event_type: &str,
    data: &str,
    blocks: &mut HashMap<usize, BlockState>,
    message_id: &mut Option<String>,
    model: &mut Option<String>,
    input_tokens: &mut u64,
    output_tokens: &mut u64,
    cache_creation_tokens: &mut Option<u64>,
    cache_read_tokens: &mut Option<u64>,
    done: &mut bool,
) -> Option<Result<ModelResponseStreamEvent, ModelError>> {
    // Parse the JSON data
    let event: StreamEvent = match serde_json::from_str(data) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("Failed to parse stream event: {} - data: {}", e, data);
            return None;
        }
    };

    match event {
        StreamEvent::MessageStart { message } => {
            *message_id = Some(message.id);
            *model = Some(message.model);
            *input_tokens = message.usage.input_tokens;
            *cache_creation_tokens = message.usage.cache_creation_input_tokens;
            *cache_read_tokens = message.usage.cache_read_input_tokens;
            None
        }

        StreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            let (state, part) = match content_block {
                ContentBlockStart::Text { text } => (
                    BlockState::Text {
                        content: text.clone(),
                    },
                    ModelResponsePart::Text(TextPart::new(&text)),
                ),
                ContentBlockStart::ToolUse { id, name, input } => (
                    BlockState::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input_json: serde_json::to_string(&input).unwrap_or_default(),
                    },
                    ModelResponsePart::ToolCall(
                        ToolCallPart::new(&name, input).with_tool_call_id(&id),
                    ),
                ),
                ContentBlockStart::Thinking { thinking } => (
                    BlockState::Thinking {
                        content: thinking.clone(),
                        signature: None,
                    },
                    ModelResponsePart::Thinking(ThinkingPart::new(&thinking)),
                ),
                ContentBlockStart::RedactedThinking { data } => (
                    BlockState::RedactedThinking {
                        signature: data.clone(),
                    },
                    ModelResponsePart::Thinking(ThinkingPart::redacted(&data, "anthropic")),
                ),
            };

            blocks.insert(index, state);
            Some(Ok(ModelResponseStreamEvent::PartStart(
                PartStartEvent::new(index, part),
            )))
        }

        StreamEvent::ContentBlockDelta { index, delta } => {
            let state = blocks.get_mut(&index)?;

            match delta {
                ContentBlockDelta::TextDelta { text } => {
                    if let BlockState::Text { content } = state {
                        content.push_str(&text);
                    }
                    Some(Ok(ModelResponseStreamEvent::PartDelta(
                        PartDeltaEvent::text(index, text),
                    )))
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    if let BlockState::ToolUse { input_json, .. } = state {
                        input_json.push_str(&partial_json);
                    }
                    Some(Ok(ModelResponseStreamEvent::PartDelta(
                        PartDeltaEvent::tool_call_args(index, partial_json),
                    )))
                }
                ContentBlockDelta::ThinkingDelta { thinking } => {
                    if let BlockState::Thinking { content, .. } = state {
                        content.push_str(&thinking);
                    }
                    Some(Ok(ModelResponseStreamEvent::PartDelta(
                        PartDeltaEvent::thinking(index, thinking),
                    )))
                }
                ContentBlockDelta::SignatureDelta { signature } => {
                    if let BlockState::Thinking { signature: sig, .. } = state {
                        match sig {
                            Some(s) => s.push_str(&signature),
                            None => *sig = Some(signature.clone()),
                        }
                    }
                    // Emit signature delta so agents can track it
                    Some(Ok(ModelResponseStreamEvent::PartDelta(PartDeltaEvent {
                        index,
                        delta: ModelResponsePartDelta::Thinking(
                            ThinkingPartDelta::new("").with_signature_delta(signature),
                        ),
                    })))
                }
            }
        }

        StreamEvent::ContentBlockStop { index } => {
            blocks.remove(&index);
            Some(Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent {
                index,
            })))
        }

        StreamEvent::MessageDelta { delta: _, usage } => {
            if let Some(u) = usage {
                *output_tokens = u.output_tokens;
            }
            // We don't emit finish reason as event since core doesn't have it
            None
        }

        StreamEvent::MessageStop => {
            *done = true;
            None
        }

        StreamEvent::Ping => None,

        StreamEvent::Error { error } => Some(Err(ModelError::api_with_code(
            error.message,
            error.error_type,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    fn make_sse_bytes(event_type: &str, data: &str) -> Bytes {
        Bytes::from(format!("event: {}\ndata: {}\n\n", event_type, data))
    }

    #[tokio::test]
    async fn test_parse_message_start() {
        let data = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let bytes = vec![Ok(make_sse_bytes("message_start", data))];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // Message start doesn't emit an event
        let event = parser.next().await;
        assert!(event.is_none());
    }

    #[tokio::test]
    async fn test_parse_text_stream() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta)),
            Ok(make_sse_bytes("content_block_stop", block_stop)),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // Collect events
        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Should have: PartStart, PartDelta, PartEnd
        assert_eq!(events.len(), 3, "Expected 3 events, got {:?}", events);

        assert!(
            matches!(&events[0], ModelResponseStreamEvent::PartStart(_)),
            "First should be PartStart"
        );
        assert!(
            matches!(&events[1], ModelResponseStreamEvent::PartDelta(_)),
            "Second should be PartDelta"
        );
        assert!(
            matches!(&events[2], ModelResponseStreamEvent::PartEnd(_)),
            "Third should be PartEnd"
        );
    }

    #[tokio::test]
    async fn test_parse_tool_use_stream() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"search","input":{}}}"#;
        let delta1 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#;
        let delta2 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta1)),
            Ok(make_sse_bytes("content_block_delta", delta2)),
            Ok(make_sse_bytes("content_block_stop", block_stop)),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Should have: PartStart, PartDelta, PartDelta, PartEnd
        assert_eq!(events.len(), 4, "Expected 4 events, got {:?}", events);

        // First should be PartStart with tool_use
        if let ModelResponseStreamEvent::PartStart(start) = &events[0] {
            assert!(
                matches!(&start.part, ModelResponsePart::ToolCall(_)),
                "Expected ToolCall"
            );
        } else {
            panic!("Expected PartStart");
        }
    }

    #[tokio::test]
    async fn test_parse_error() {
        let error =
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#;
        let bytes = vec![Ok(make_sse_bytes("error", error))];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let event = parser.next().await.unwrap();
        assert!(event.is_err());
        let err = event.unwrap_err();
        assert!(err.to_string().contains("Rate limited"));
    }

    #[tokio::test]
    async fn test_parse_ping() {
        let ping = r#"{"type":"ping"}"#;
        let bytes = vec![Ok(make_sse_bytes("ping", ping))];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // Ping shouldn't emit an event
        let event = parser.next().await;
        assert!(event.is_none());
    }
}
