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
