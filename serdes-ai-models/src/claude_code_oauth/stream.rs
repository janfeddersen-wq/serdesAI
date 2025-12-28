//! Claude Code OAuth SSE stream parser.
//!
//! This module reuses the Anthropic SSE parsing since Claude Code
//! uses the same API format.

use crate::anthropic::stream::AnthropicStreamParser;
use crate::error::ModelError;
use bytes::Bytes;
use futures::Stream;
use serdes_ai_core::messages::ModelResponseStreamEvent;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Claude Code SSE stream wrapper.
/// 
/// This is a thin wrapper around `AnthropicStreamParser` since
/// Claude Code uses the same Anthropic Messages API.
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
}

impl<S> Stream for ClaudeCodeStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}
