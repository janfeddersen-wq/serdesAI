//! OpenAI SSE stream parser.
//!
//! This module provides streaming support for OpenAI chat completions.

use super::types::ChatCompletionChunk;
use crate::error::ModelError;
use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;
use serdes_ai_core::messages::{
    ModelResponseStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent, TextPart, ThinkingPart,
    ThinkingPartDelta, ToolCallArgs, ToolCallPart,
};
use serdes_ai_core::ModelResponsePart;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    /// OpenAI SSE stream parser.
    pub struct OpenAIStreamParser<S> {
        #[pin]
        inner: S,
        buffer: String,
        // Track tool calls in progress (index -> accumulated data)
        tool_calls: HashMap<u32, ToolCallState>,
        // Track text part state
        text_started: bool,
        // Track if we've emitted a part start for text
        current_text_index: Option<usize>,
        // Track thinking/reasoning part state
        thinking_started: bool,
        // Track if we've emitted a part start for thinking
        current_thinking_index: Option<usize>,
        // Next part index to use
        next_part_index: usize,
        // Finished
        done: bool,
        // Pending PartEnd events to emit (queued when multiple parts close at once)
        pending_part_ends: Vec<usize>,
    }
}

/// State for an in-progress tool call.
#[derive(Debug, Clone, Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
    part_index: usize,
    started: bool,
}

impl<S> OpenAIStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    /// Create a new stream parser.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            tool_calls: HashMap::new(),
            text_started: false,
            current_text_index: None,
            thinking_started: false,
            current_thinking_index: None,
            next_part_index: 0,
            done: false,
            pending_part_ends: Vec::new(),
        }
    }
}

impl<S> Stream for OpenAIStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.done {
            return Poll::Ready(None);
        }

        // First, emit any pending PartEnd events
        if let Some(idx) = this.pending_part_ends.pop() {
            return Poll::Ready(Some(Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent {
                index: idx,
            }))));
        }

        loop {
            // Check if we have complete lines in the buffer
            while let Some(newline_pos) = this.buffer.find('\n') {
                let line = this.buffer.drain(..=newline_pos).collect::<String>();
                if let Some(event) = parse_sse_line(
                    &line,
                    this.tool_calls,
                    this.text_started,
                    this.current_text_index,
                    this.thinking_started,
                    this.current_thinking_index,
                    this.next_part_index,
                    this.done,
                    this.pending_part_ends,
                ) {
                    return Poll::Ready(Some(event));
                }
            }

            // Need more data
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        this.buffer.push_str(text);
                    }
                    // Continue to process buffer
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ModelError::Other(e.into()))));
                }
                Poll::Ready(None) => {
                    // Stream ended, process any remaining buffer
                    if !this.buffer.is_empty() {
                        let remaining = std::mem::take(this.buffer);
                        for line in remaining.lines() {
                            if let Some(event) = parse_sse_line(
                                line,
                                this.tool_calls,
                                this.text_started,
                                this.current_text_index,
                                this.thinking_started,
                                this.current_thinking_index,
                                this.next_part_index,
                                this.done,
                                this.pending_part_ends,
                            ) {
                                return Poll::Ready(Some(event));
                            }
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Parse a single SSE line into an event.
fn parse_sse_line(
    line: &str,
    tool_calls: &mut HashMap<u32, ToolCallState>,
    text_started: &mut bool,
    current_text_index: &mut Option<usize>,
    thinking_started: &mut bool,
    current_thinking_index: &mut Option<usize>,
    next_part_index: &mut usize,
    done: &mut bool,
    pending_part_ends: &mut Vec<usize>,
) -> Option<Result<ModelResponseStreamEvent, ModelError>> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return None;
    }

    // Handle data lines
    if let Some(data) = line.strip_prefix("data: ") {
        // Check for stream end
        if data == "[DONE]" {
            *done = true;
            return None;
        }

        // Parse the JSON chunk
        match serde_json::from_str::<ChatCompletionChunk>(data) {
            Ok(chunk) => {
                // Process each choice
                for choice in chunk.choices {
                    let delta = choice.delta;

                    // Handle text content
                    if let Some(content) = delta.content {
                        if !content.is_empty() {
                            // Start text part if not started
                            if !*text_started {
                                *text_started = true;
                                *current_text_index = Some(*next_part_index);
                                *next_part_index += 1;

                                // Include the first content in the PartStart
                                let start = PartStartEvent::new(
                                    current_text_index.unwrap(),
                                    ModelResponsePart::Text(TextPart::new(&content)),
                                );
                                return Some(Ok(ModelResponseStreamEvent::PartStart(start)));
                            }

                            // Emit text delta for subsequent content
                            let delta = PartDeltaEvent::text(current_text_index.unwrap(), content);
                            return Some(Ok(ModelResponseStreamEvent::PartDelta(delta)));
                        }
                    }

                    // Handle reasoning/thinking content (for models like GLM-4)
                    if let Some(reasoning) = delta.reasoning_content {
                        if !reasoning.is_empty() {
                            // Start thinking part if not started
                            if !*thinking_started {
                                *thinking_started = true;
                                *current_thinking_index = Some(*next_part_index);
                                *next_part_index += 1;

                                // Include the first content in the PartStart
                                let start = PartStartEvent::new(
                                    current_thinking_index.unwrap(),
                                    ModelResponsePart::Thinking(ThinkingPart::new(&reasoning)),
                                );
                                return Some(Ok(ModelResponseStreamEvent::PartStart(start)));
                            }

                            // Emit thinking delta for subsequent content
                            let delta_event = PartDeltaEvent {
                                index: current_thinking_index.unwrap(),
                                delta: serdes_ai_core::messages::ModelResponsePartDelta::Thinking(
                                    ThinkingPartDelta::new(reasoning),
                                ),
                            };
                            return Some(Ok(ModelResponseStreamEvent::PartDelta(delta_event)));
                        }
                    }

                    // Handle tool calls
                    if let Some(tcs) = delta.tool_calls {
                        for tc in tcs {
                            let state = tool_calls.entry(tc.index).or_insert_with(|| {
                                let part_index = *next_part_index;
                                *next_part_index += 1;
                                ToolCallState {
                                    part_index,
                                    ..Default::default()
                                }
                            });

                            // Update state
                            if let Some(id) = tc.id {
                                state.id = id;
                            }
                            if let Some(func) = tc.function {
                                if let Some(name) = func.name {
                                    state.name = name;
                                }
                                if let Some(args) = func.arguments {
                                    state.arguments.push_str(&args);

                                    // Start part if not started
                                    if !state.started {
                                        state.started = true;
                                        // Include the accumulated args in PartStart (some providers
                                        // like Cerebras send all tool call data in one chunk)
                                        let tool_part = ToolCallPart::new(
                                            &state.name,
                                            ToolCallArgs::String(state.arguments.clone()),
                                        )
                                        .with_tool_call_id(&state.id);

                                        let start = PartStartEvent::new(
                                            state.part_index,
                                            ModelResponsePart::ToolCall(tool_part),
                                        );
                                        return Some(Ok(ModelResponseStreamEvent::PartStart(
                                            start,
                                        )));
                                    }

                                    // Emit args delta
                                    let delta =
                                        PartDeltaEvent::tool_call_args(state.part_index, args);
                                    return Some(Ok(ModelResponseStreamEvent::PartDelta(delta)));
                                }
                            }
                        }
                    }

                    // Handle finish reason - emit part end events
                    if choice.finish_reason.is_some() {
                        // Collect all part indices that need to be closed
                        let mut parts_to_close = Vec::new();

                        // End text part if open
                        if let Some(idx) = current_text_index.take() {
                            parts_to_close.push(idx);
                        }

                        // End tool call parts
                        for (_, state) in tool_calls.drain() {
                            if state.started {
                                parts_to_close.push(state.part_index);
                            }
                        }

                        // If we have parts to close, return the first one and queue the rest
                        if !parts_to_close.is_empty() {
                            // Queue all but the first for later emission
                            if parts_to_close.len() > 1 {
                                pending_part_ends.extend(parts_to_close.iter().skip(1).copied());
                            }
                            // Return the first one now
                            let first_idx = parts_to_close[0];
                            return Some(Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent {
                                index: first_idx,
                            })));
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to parse SSE chunk: {} - data: {}", e, data);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;
    use serdes_ai_core::ModelResponsePartDelta;

    fn make_chunk_bytes(data: &str) -> Bytes {
        Bytes::from(format!("data: {}\n\n", data))
    }

    #[tokio::test]
    async fn test_parse_text_chunk() {
        // Test multi-chunk text streaming
        let chunk1 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#;
        let chunk2 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
        let chunk3 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" World"}}]}"#;
        let bytes = vec![
            Ok(make_chunk_bytes(chunk1)),
            Ok(make_chunk_bytes(chunk2)),
            Ok(make_chunk_bytes(chunk3)),
        ];
        let stream = stream::iter(bytes);
        let mut parser = OpenAIStreamParser::new(stream);

        // Collect all events
        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Should have: PartStart(Hello), PartDelta( World)
        assert!(
            events.len() >= 2,
            "Expected at least 2 events, got {}: {:?}",
            events.len(),
            events
        );

        // First should be PartStart with "Hello" content
        if let ModelResponseStreamEvent::PartStart(start) = &events[0] {
            if let ModelResponsePart::Text(text) = &start.part {
                assert_eq!(text.content, "Hello", "PartStart should contain 'Hello'");
            } else {
                panic!("Expected Text part in PartStart, got {:?}", start.part);
            }
        } else {
            panic!("First event should be PartStart, got {:?}", events[0]);
        }

        // Second should be PartDelta with " World"
        if let ModelResponseStreamEvent::PartDelta(delta) = &events[1] {
            if let ModelResponsePartDelta::Text(text) = &delta.delta {
                assert_eq!(
                    text.content_delta, " World",
                    "Delta should contain ' World'"
                );
            } else {
                panic!("Expected Text delta, got {:?}", delta.delta);
            }
        } else {
            panic!("Second event should be PartDelta, got {:?}", events[1]);
        }
    }

    #[tokio::test]
    async fn test_parse_done() {
        let bytes = vec![Ok(Bytes::from("data: [DONE]\n\n"))];
        let stream = stream::iter(bytes);
        let mut parser = OpenAIStreamParser::new(stream);

        // Should return None (stream ended)
        let event = parser.next().await;
        assert!(event.is_none());
    }

    #[tokio::test]
    async fn test_parse_finish_reason() {
        let chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let bytes = vec![Ok(make_chunk_bytes(chunk))];
        let stream = stream::iter(bytes);
        let mut parser = OpenAIStreamParser::new(stream);

        // Should return None since no parts are open
        let event = parser.next().await;
        assert!(event.is_none());
    }

    #[tokio::test]
    async fn test_parse_tool_call() {
        let chunk1 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]}}]}"#;
        let chunk2 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}"#;
        let chunk3 = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"test\"}"}}]}}]}"#;

        let bytes = vec![
            Ok(make_chunk_bytes(chunk1)),
            Ok(make_chunk_bytes(chunk2)),
            Ok(make_chunk_bytes(chunk3)),
        ];
        let stream = stream::iter(bytes);
        let mut parser = OpenAIStreamParser::new(stream);

        // First event should be PartStart for tool call
        let event = parser.next().await.unwrap().unwrap();
        assert!(matches!(event, ModelResponseStreamEvent::PartStart(_)));

        // Subsequent events should be deltas
        let event = parser.next().await.unwrap().unwrap();
        assert!(matches!(event, ModelResponseStreamEvent::PartDelta(_)));
    }

    /// Regression test: when finish_reason is received with multiple open parts
    /// (text + tool calls), ALL PartEnd events must be emitted, not just the first.
    #[tokio::test]
    async fn test_multiple_part_ends_on_finish() {
        // Scenario: text part starts, then 2 tool calls start, then finish_reason
        // We should get 3 PartEnd events (one for text, two for tool calls)
        let text_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
        let tool1_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":\"test\"}"}}]}}]}"#;
        let tool2_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"id\":1}"}}]}}]}"#;
        let finish_chunk = r#"{"id":"123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;

        let bytes = vec![
            Ok(make_chunk_bytes(text_chunk)),
            Ok(make_chunk_bytes(tool1_chunk)),
            Ok(make_chunk_bytes(tool2_chunk)),
            Ok(make_chunk_bytes(finish_chunk)),
        ];
        let stream = stream::iter(bytes);
        let mut parser = OpenAIStreamParser::new(stream);

        // Collect all events
        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Should have:
        // - 1 PartStart for text (index 0)
        // - 1 PartStart for tool call 1 (index 1)
        // - 1 PartStart for tool call 2 (index 2)
        // - 3 PartEnd events (for indices 0, 1, 2)
        let part_starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ModelResponseStreamEvent::PartStart(_)))
            .collect();
        let part_ends: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ModelResponseStreamEvent::PartEnd(_)))
            .collect();

        assert_eq!(
            part_starts.len(),
            3,
            "Expected 3 PartStart events, got {}: {:?}",
            part_starts.len(),
            part_starts
        );
        assert_eq!(
            part_ends.len(),
            3,
            "Expected 3 PartEnd events (regression: bug caused only 1 to be emitted), got {}: {:?}",
            part_ends.len(),
            part_ends
        );

        // Verify all part indices are closed
        let mut closed_indices: Vec<usize> = part_ends
            .iter()
            .filter_map(|e| {
                if let ModelResponseStreamEvent::PartEnd(end) = e {
                    Some(end.index)
                } else {
                    None
                }
            })
            .collect();
        closed_indices.sort();
        assert_eq!(
            closed_indices,
            vec![0, 1, 2],
            "All part indices should be closed"
        );
    }
}
