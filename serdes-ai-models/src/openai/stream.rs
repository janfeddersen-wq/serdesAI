//! OpenAI-compatible Chat SSE parser. `[DONE]` is the transport terminal marker.
//! Finish frames alone are not terminal: a usage-only frame may follow them.
use super::types::{ChatCompletionChunk, Usage};
use crate::error::ModelError;
use bytes::Bytes;
use futures::Stream;
use serdes_ai_core::ModelResponsePart;
use serdes_ai_core::messages::{
    FinishReason, ModelResponseStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent,
    StreamCompleteEvent, TextPart, ThinkingPart, ThinkingPartDelta, ToolCallArgs, ToolCallPart,
};
use std::{
    collections::{BTreeMap, VecDeque},
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
    part_index: usize,
    started: bool,
}

/// Incremental parser with byte buffering (HTTP chunks need not be UTF-8 boundaries).
pub struct OpenAIStreamParser<S> {
    inner: Pin<Box<S>>,
    buffer: Vec<u8>,
    pending: VecDeque<Result<ModelResponseStreamEvent, ModelError>>,
    tools: BTreeMap<u32, ToolCallState>,
    text: Option<usize>,
    thinking: Option<usize>,
    next_index: usize,
    finish: Option<FinishReason>,
    usage: Option<Usage>,
    done: bool,
    finish_is_terminal: bool,
}

impl<S> OpenAIStreamParser<S> {
    /// Create a parser for a Chat Completions endpoint using the `[DONE]` contract.
    pub fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
            buffer: Vec::new(),
            pending: VecDeque::new(),
            tools: BTreeMap::new(),
            text: None,
            thinking: None,
            next_index: 0,
            finish: None,
            usage: None,
            done: false,
            finish_is_terminal: false,
        }
    }

    /// Allow clean EOF after a finish frame for compatible endpoints without `[DONE]`.
    /// Disabled by default. Usage frames are still consumed until EOF.
    pub fn with_finish_reason_terminal(mut self, enabled: bool) -> Self {
        self.finish_is_terminal = enabled;
        self
    }

    fn fail(&mut self, error: ModelError) {
        self.done = true;
        self.pending.push_back(Err(error));
    }

    fn emit(&mut self, event: ModelResponseStreamEvent) {
        self.pending.push_back(Ok(event));
    }

    fn validate_tools(&mut self) -> bool {
        if self.finish == Some(FinishReason::Length) {
            return true;
        }
        if self.tools.values().any(|tool| {
            !tool.started || serde_json::from_str::<serde_json::Value>(&tool.arguments).is_err()
        }) {
            self.fail(ModelError::invalid_response(
                "Incomplete Chat tool call at terminal frame",
            ));
            return false;
        }
        true
    }

    fn close_parts(&mut self) {
        let mut indices: Vec<_> = self
            .text
            .take()
            .into_iter()
            .chain(self.thinking.take())
            .collect();
        indices.extend(
            std::mem::take(&mut self.tools)
                .into_values()
                .filter(|tool| tool.started)
                .map(|tool| tool.part_index),
        );
        indices.sort_unstable();
        for index in indices {
            self.emit(ModelResponseStreamEvent::PartEnd(PartEndEvent { index }));
        }
    }

    fn line(&mut self, bytes: &[u8]) {
        let Ok(line) = std::str::from_utf8(bytes) else {
            self.fail(ModelError::invalid_response("Invalid UTF-8 in Chat SSE"));
            return;
        };
        let Some(data) = line.trim().strip_prefix("data:") else {
            return;
        };
        let data = data.trim_start();
        if data == "[DONE]" {
            if !self.validate_tools() {
                return;
            }
            self.close_parts();
            self.emit(stream_complete_event(self.finish, self.usage.as_ref()));
            self.done = true;
            return;
        }
        let value: serde_json::Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => {
                self.fail(ModelError::invalid_response("Malformed Chat SSE JSON"));
                return;
            }
        };
        if let Some(error) = value.get("error") {
            self.fail(ModelError::Api {
                message: error
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Chat provider stream error")
                    .to_owned(),
                code: error
                    .get("code")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
            });
            return;
        }
        let chunk: ChatCompletionChunk = match serde_json::from_value(value) {
            Ok(chunk) => chunk,
            Err(_) => {
                self.fail(ModelError::invalid_response(
                    "Invalid Chat SSE chunk schema",
                ));
                return;
            }
        };
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        // This API returns one ModelResponse; alternative choices must not be concatenated.
        for choice in chunk.choices.into_iter().filter(|choice| choice.index == 0) {
            if self.finish.is_some() {
                self.fail(ModelError::invalid_response(
                    "Chat choice after finish frame",
                ));
                return;
            }
            let delta = choice.delta;
            if let Some(content) = delta.content.filter(|s| !s.is_empty()) {
                if let Some(index) = self.text {
                    self.emit(ModelResponseStreamEvent::PartDelta(PartDeltaEvent::text(
                        index, content,
                    )));
                } else {
                    let index = self.next_index;
                    self.next_index += 1;
                    self.text = Some(index);
                    self.emit(ModelResponseStreamEvent::PartStart(PartStartEvent::new(
                        index,
                        ModelResponsePart::Text(TextPart::new(content)),
                    )));
                }
            }
            // Compatibility extension only; OpenAI Responses reasoning uses a different schema.
            if let Some(content) = delta.reasoning_content.filter(|s| !s.is_empty()) {
                if let Some(index) = self.thinking {
                    self.emit(ModelResponseStreamEvent::PartDelta(PartDeltaEvent {
                        index,
                        delta: serdes_ai_core::ModelResponsePartDelta::Thinking(
                            ThinkingPartDelta::new(content),
                        ),
                    }));
                } else {
                    let index = self.next_index;
                    self.next_index += 1;
                    self.thinking = Some(index);
                    self.emit(ModelResponseStreamEvent::PartStart(PartStartEvent::new(
                        index,
                        ModelResponsePart::Thinking(ThinkingPart::new(content)),
                    )));
                }
            }
            for tool in delta.tool_calls.unwrap_or_default() {
                let state = self.tools.entry(tool.index).or_insert_with(|| {
                    let part_index = self.next_index;
                    self.next_index += 1;
                    ToolCallState {
                        part_index,
                        ..Default::default()
                    }
                });
                if let Some(id) = tool.id {
                    state.id = id;
                }
                if let Some(function) = tool.function {
                    if let Some(name) = function.name {
                        state.name.push_str(&name);
                    }
                    if let Some(args) = function.arguments {
                        state.arguments.push_str(&args);
                        if state.started {
                            let event = PartDeltaEvent::tool_call_args(state.part_index, args);
                            self.pending
                                .push_back(Ok(ModelResponseStreamEvent::PartDelta(event)));
                        }
                    }
                }
                if !state.started && !state.id.is_empty() && !state.name.is_empty() {
                    state.started = true;
                    let part = ToolCallPart::new(
                        &state.name,
                        ToolCallArgs::String(state.arguments.clone()),
                    )
                    .with_tool_call_id(&state.id);
                    self.pending
                        .push_back(Ok(ModelResponseStreamEvent::PartStart(
                            PartStartEvent::new(
                                state.part_index,
                                ModelResponsePart::ToolCall(part),
                            ),
                        )));
                }
            }
            if let Some(reason) = choice.finish_reason {
                self.finish = Some(map_finish_reason(&reason));
                if !self.validate_tools() {
                    return;
                }
                self.close_parts();
            }
        }
    }
}

impl<S: Stream<Item = Result<Bytes, reqwest::Error>>> Stream for OpenAIStreamParser<S> {
    type Item = Result<ModelResponseStreamEvent, ModelError>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(event));
            }
            if this.done {
                return Poll::Ready(None);
            }
            if let Some(end) = this.buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<_> = this.buffer.drain(..=end).collect();
                this.line(&line);
                continue;
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => this.buffer.extend_from_slice(&bytes),
                Poll::Ready(Some(Err(error))) => this.fail(ModelError::Other(error.into())),
                Poll::Ready(None) => {
                    if !this.buffer.is_empty() {
                        let line = std::mem::take(&mut this.buffer);
                        this.line(&line);
                    }
                    if !this.done && this.finish_is_terminal && this.finish.is_some() {
                        this.emit(stream_complete_event(this.finish, this.usage.as_ref()));
                        this.done = true;
                    }
                    if !this.done {
                        this.fail(ModelError::incomplete_stream(
                            "Chat SSE ended before [DONE]",
                        ));
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Build the terminal event from the last seen finish reason and the buffered
/// usage chunk; fields absent from the wire stay `None`.
fn stream_complete_event(
    finish_reason: Option<FinishReason>,
    usage: Option<&Usage>,
) -> ModelResponseStreamEvent {
    let reason = finish_reason.unwrap_or(FinishReason::Stop);
    let mut event = StreamCompleteEvent::new(reason);

    if let Some(u) = usage {
        event = event
            .with_input_tokens(u.prompt_tokens)
            .with_output_tokens(u.completion_tokens);
        // The OpenAI wire format reports cached prompt tokens but has no
        // cache-creation count, so cache_creation_tokens stays None.
        if let Some(cached) = u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
        {
            event = event.with_cache_read_tokens(cached);
        }
    }

    ModelResponseStreamEvent::StreamComplete(event)
}

/// Map an OpenAI finish reason string to a [`FinishReason`].
///
/// Unknown reasons map to [`FinishReason::Stop`], matching the non-streaming
/// response mapping.
fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        "tool_calls" => FinishReason::ToolCall,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
