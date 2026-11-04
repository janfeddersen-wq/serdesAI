//! Claude Code OAuth SSE stream parser.
//!
//! This module wraps the Anthropic SSE parsing and adds tool name
//! unprefixing for Claude Code OAuth compatibility.

use crate::anthropic::stream::AnthropicStreamParser;
use crate::error::ModelError;
use bytes::Bytes;
use futures::Stream;
use serdes_ai_core::messages::{ModelResponsePart, ModelResponseStreamEvent};
use std::pin::Pin;
use std::task::{Context, Poll};

/// Tool name prefix that must be stripped from responses.
const TOOL_PREFIX: &str = "cp_";

/// Claude Code SSE stream wrapper.
///
/// This wraps `AnthropicStreamParser` and adds tool name unprefixing
/// since Claude Code OAuth requires the `cp_` prefix on outgoing tools
/// but we need to strip it from incoming responses.
pub struct ClaudeCodeStreamParser<S> {
    inner: AnthropicStreamParser<S>,
}

impl<S> ClaudeCodeStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    /// Create a new stream parser from a byte stream.
    pub fn new(byte_stream: S) -> Self {
        Self {
            inner: AnthropicStreamParser::new(byte_stream),
        }
    }

    /// Strip the cp_ prefix from tool names in a stream event.
    fn unprefix_tool_names(event: ModelResponseStreamEvent) -> ModelResponseStreamEvent {
        match event {
            ModelResponseStreamEvent::PartStart(mut start_event) => {
                // Check if this is a tool call and strip the prefix
                if let ModelResponsePart::ToolCall(ref mut tc) = start_event.part {
                    if let Some(unprefixed) = tc.tool_name.strip_prefix(TOOL_PREFIX) {
                        tc.tool_name = unprefixed.to_string();
                    }
                }
                ModelResponseStreamEvent::PartStart(start_event)
            }
            // Pass through all other events unchanged
            other => other,
        }
    }
}

impl<S> Stream for ClaudeCodeStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(Ok(Self::unprefix_tool_names(event)))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;
    use serdes_ai_core::messages::FinishReason;

    fn make_sse_bytes(event_type: &str, data: &str) -> Bytes {
        Bytes::from(format!("event: {}\ndata: {}\n\n", event_type, data))
    }

    /// The OAuth adapter inherits the refusal mapping of the Anthropic parser.
    #[tokio::test]
    async fn refusal_reaches_the_terminal_event_with_its_category() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"reasoning_extraction","explanation":"private text"}},"usage":{"output_tokens":1}}"#;
        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("message_delta", msg_delta)),
            Ok(make_sse_bytes("message_stop", r#"{"type":"message_stop"}"#)),
        ];

        let events: Vec<_> = ClaudeCodeStreamParser::new(stream::iter(bytes))
            .collect()
            .await;

        match events.last() {
            Some(Ok(ModelResponseStreamEvent::StreamComplete(complete))) => {
                assert_eq!(complete.finish_reason, FinishReason::ContentFilter);
                let details = complete
                    .metadata
                    .as_ref()
                    .unwrap()
                    .details
                    .as_ref()
                    .unwrap();
                let stop = crate::refusal::StopDetails::from_details(details).unwrap();
                assert_eq!(stop.category.as_deref(), Some("reasoning_extraction"));
                assert!(!details.to_string().contains("private text"));
            }
            other => panic!("expected terminal StreamComplete last, got {:?}", other),
        }
    }

    /// The wrapper passes the wrapped Anthropic parser's terminal
    /// StreamComplete through unchanged: exactly one terminal event,
    /// emitted last, with the usage fields the Anthropic stream reported.
    #[tokio::test]
    async fn passes_through_terminal_stream_complete() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":7}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta)),
            Ok(make_sse_bytes("content_block_stop", block_stop)),
            Ok(make_sse_bytes("message_delta", msg_delta)),
            Ok(make_sse_bytes("message_stop", msg_stop)),
        ];

        let mut parser = ClaudeCodeStreamParser::new(stream::iter(bytes));

        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Part events precede the terminal event.
        assert_eq!(
            events.len(),
            4,
            "Expected PartStart, PartDelta, PartEnd, StreamComplete, got {:?}",
            events
        );

        let terminals: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ModelResponseStreamEvent::StreamComplete(_)))
            .collect();
        assert_eq!(terminals.len(), 1, "expected exactly one terminal event");

        match events.last() {
            Some(ModelResponseStreamEvent::StreamComplete(complete)) => {
                assert_eq!(complete.finish_reason, FinishReason::EndTurn);
                assert_eq!(complete.input_tokens, Some(10));
                assert_eq!(complete.output_tokens, Some(5));
                assert_eq!(complete.cache_creation_tokens, Some(3));
                assert_eq!(complete.cache_read_tokens, Some(7));
            }
            other => panic!("expected terminal StreamComplete last, got {:?}", other),
        }
    }
}
