//! Antigravity SSE stream parser.
//!
//! Parses Server-Sent Events from the Antigravity API into model responses.

use super::types::*;
use crate::error::ModelError;
use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;
use serdes_ai_core::messages::{
    ModelResponseStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent, TextPart, ThinkingPart,
    ToolCallPart,
};
use serdes_ai_core::ModelResponsePart;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{trace, warn};

pin_project! {
    /// SSE stream parser for Antigravity responses.
    pub struct AntigravityStreamParser<S> {
        #[pin]
        inner: S,
        buffer: String,
        // Track parts in progress
        parts: HashMap<usize, PartState>,
        // Current part index
        next_part_index: usize,
        // Finished
        done: bool,
    }
}

/// State for an in-progress part.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum PartState {
    Text { content: String },
    FunctionCall { name: String, args: String },
    Thinking { content: String },
}

impl<S> AntigravityStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    /// Create a new stream parser.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            parts: HashMap::new(),
            next_part_index: 0,
            done: false,
        }
    }
}

impl<S> Stream for AntigravityStreamParser<S>
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
            // Try to parse complete SSE events from buffer
            while let Some(event_end) = this.buffer.find("\n\n") {
                let event = this.buffer.drain(..event_end + 2).collect::<String>();

                // Parse SSE event lines
                for line in event.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        // Handle [DONE] marker
                        if data == "[DONE]" {
                            *this.done = true;
                            return Poll::Ready(None);
                        }

                        // Parse the JSON response
                        match serde_json::from_str::<AntigravityResponse>(data) {
                            Ok(response) => {
                                if let Some(event) = process_response(
                                    &response,
                                    this.parts,
                                    this.next_part_index,
                                    this.done,
                                ) {
                                    return Poll::Ready(Some(event));
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse Antigravity stream chunk: {}", e);
                                trace!("Raw data: {}", data);
                            }
                        }
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
                    // Process any remaining buffer
                    if !this.buffer.is_empty() {
                        let remaining = std::mem::take(this.buffer);
                        for line in remaining.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data != "[DONE]" {
                                    if let Ok(response) =
                                        serde_json::from_str::<AntigravityResponse>(data)
                                    {
                                        if let Some(event) = process_response(
                                            &response,
                                            this.parts,
                                            this.next_part_index,
                                            this.done,
                                        ) {
                                            return Poll::Ready(Some(event));
                                        }
                                    }
                                }
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

/// Process a response chunk into stream events.
fn process_response(
    response: &AntigravityResponse,
    parts: &mut HashMap<usize, PartState>,
    next_part_index: &mut usize,
    done: &mut bool,
) -> Option<Result<ModelResponseStreamEvent, ModelError>> {
    // Get the first candidate
    let candidate = response.candidates.first()?;
    let content = candidate.content.as_ref()?;

    // Process each part
    for part in &content.parts {
        match part {
            Part::Text { text } => {
                if text.is_empty() {
                    continue;
                }

                // Check if we have an existing text part
                let text_part_idx = parts.iter().find_map(|(idx, state)| {
                    if matches!(state, PartState::Text { .. }) {
                        Some(*idx)
                    } else {
                        None
                    }
                });

                if let Some(idx) = text_part_idx {
                    // Emit delta for existing part
                    if let Some(PartState::Text { content }) = parts.get_mut(&idx) {
                        let delta = text.clone();
                        content.push_str(&delta);
                        return Some(Ok(ModelResponseStreamEvent::PartDelta(
                            PartDeltaEvent::text(idx, delta),
                        )));
                    }
                } else {
                    // Start new text part
                    let idx = *next_part_index;
                    *next_part_index += 1;
                    parts.insert(
                        idx,
                        PartState::Text {
                            content: text.clone(),
                        },
                    );
                    return Some(Ok(ModelResponseStreamEvent::PartStart(
                        PartStartEvent::new(idx, ModelResponsePart::Text(TextPart::new(text))),
                    )));
                }
            }
            Part::FunctionCall { function_call } => {
                // Start new function call part
                let idx = *next_part_index;
                *next_part_index += 1;

                let mut tool_part =
                    ToolCallPart::new(&function_call.name, function_call.args.clone());
                if let Some(id) = &function_call.id {
                    tool_part = tool_part.with_tool_call_id(id);
                }

                parts.insert(
                    idx,
                    PartState::FunctionCall {
                        name: function_call.name.clone(),
                        args: serde_json::to_string(&function_call.args).unwrap_or_default(),
                    },
                );

                return Some(Ok(ModelResponseStreamEvent::PartStart(
                    PartStartEvent::new(idx, ModelResponsePart::ToolCall(tool_part)),
                )));
            }
            Part::Thinking { thought, .. } => {
                if thought.is_empty() {
                    continue;
                }

                // Check if we have an existing thinking part
                let think_part_idx = parts.iter().find_map(|(idx, state)| {
                    if matches!(state, PartState::Thinking { .. }) {
                        Some(*idx)
                    } else {
                        None
                    }
                });

                if let Some(idx) = think_part_idx {
                    if let Some(PartState::Thinking { content }) = parts.get_mut(&idx) {
                        let delta = thought.clone();
                        content.push_str(&delta);
                        return Some(Ok(ModelResponseStreamEvent::PartDelta(
                            PartDeltaEvent::thinking(idx, delta),
                        )));
                    }
                } else {
                    let idx = *next_part_index;
                    *next_part_index += 1;
                    parts.insert(
                        idx,
                        PartState::Thinking {
                            content: thought.clone(),
                        },
                    );
                    return Some(Ok(ModelResponseStreamEvent::PartStart(
                        PartStartEvent::new(
                            idx,
                            ModelResponsePart::Thinking(ThinkingPart::new(thought)),
                        ),
                    )));
                }
            }
            _ => {}
        }
    }

    // Check for finish
    if candidate.finish_reason.is_some() {
        *done = true;

        // Emit end events for all parts
        if let Some((idx, _)) = parts.drain().next() {
            return Some(Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent {
                index: idx,
            })));
        }
    }

    None
}
