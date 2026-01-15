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
use tracing::{debug, trace, warn};

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
            debug!("Buffer length: {}, looking for events", this.buffer.len());
            if this.buffer.len() > 0 && this.buffer.len() < 1000 {
                debug!("Buffer content: {:?}", this.buffer);
            } else if this.buffer.len() >= 1000 {
                debug!("Buffer first 500 chars: {:?}", &this.buffer[..500]);
            }
            // Try both \r\n\r\n (Windows) and \n\n (Unix) line endings
            let separator = if this.buffer.contains("\r\n\r\n") {
                "\r\n\r\n"
            } else {
                "\n\n"
            };
            while let Some(event_end) = this.buffer.find(separator) {
                let event = this.buffer.drain(..event_end + separator.len()).collect::<String>();
                debug!("Found SSE event: {:?}", &event[..event.len().min(200)]);

                // Parse SSE event lines
                for line in event.lines() {
                    debug!("SSE line: {:?}", line);
                    if let Some(data) = line.strip_prefix("data: ") {
                        debug!("SSE data: {:?}", &data[..data.len().min(200)]);
                        // Handle [DONE] marker
                        if data == "[DONE]" {
                            debug!("Got [DONE] marker");
                            *this.done = true;
                            return Poll::Ready(None);
                        }

                        // Parse the JSON response
                        match serde_json::from_str::<AntigravityResponse>(data) {
                            Ok(response) => {
                                debug!("Parsed response with {} candidates", response.response.candidates.len());
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
                                debug!("Raw data that failed: {}", data);
                            }
                        }
                    }
                }
            }

            // Need more data
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        trace!("SSE chunk received: {} bytes", text.len());
                        trace!("SSE chunk content: {:?}", &text[..text.len().min(500)]);
                        this.buffer.push_str(text);
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ModelError::Other(e.into()))));
                }
                Poll::Ready(None) => {
                    *this.done = true;
                    debug!("Stream ended, remaining buffer length: {}", this.buffer.len());
                    if !this.buffer.is_empty() {
                        debug!("Final buffer content (first 1000): {:?}", &this.buffer[..this.buffer.len().min(1000)]);
                    }
                    // Process any remaining buffer - try different line endings
                    if !this.buffer.is_empty() {
                        let remaining = std::mem::take(this.buffer);
                        // Try splitting by single \n as well
                        for line in remaining.lines() {
                            debug!("Processing remaining line: {:?}", &line[..line.len().min(200)]);
                            if let Some(data) = line.strip_prefix("data: ") {
                                debug!("Found data line: {:?}", &data[..data.len().min(200)]);
                                if data != "[DONE]" {
                                    match serde_json::from_str::<AntigravityResponse>(data) {
                                        Ok(response) => {
                                            debug!("Parsed final response with {} candidates", response.response.candidates.len());
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
                                            debug!("Failed to parse final data: {}", e);
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
    // Get the first candidate from the wrapped response
    let candidate = response.response.candidates.first()?;
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
            Part::FunctionCall { function_call, thought_signature } => {
                // Start new function call part
                let idx = *next_part_index;
                *next_part_index += 1;

                let mut tool_part =
                    ToolCallPart::new(&function_call.name, function_call.args.clone());
                if let Some(id) = &function_call.id {
                    tool_part = tool_part.with_tool_call_id(id);
                }
                
                // Store thought signature in provider_details for multi-turn tool calls
                if let Some(sig) = thought_signature {
                    let mut details = serde_json::Map::new();
                    details.insert("thoughtSignature".to_string(), serde_json::Value::String(sig.clone()));
                    tool_part.provider_details = Some(details);
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
            Part::Thinking { thought: _, text } => {
                // thought is a bool flag, text contains the actual thinking content
                if text.is_empty() {
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
                        let delta = text.clone();
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
                            content: text.clone(),
                        },
                    );
                    return Some(Ok(ModelResponseStreamEvent::PartStart(
                        PartStartEvent::new(
                            idx,
                            ModelResponsePart::Thinking(ThinkingPart::new(text)),
                        ),
                    )));
                }
            }
            Part::ThoughtSignature { .. } => {
                // Thought signatures are used for multi-turn, we can skip them for now
                continue;
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
