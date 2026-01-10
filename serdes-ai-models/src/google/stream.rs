//! Google AI SSE stream parser.
//!
//! Google uses a slightly different streaming format - each chunk is a complete
//! JSON response object, not deltas.

use super::types::{GenerateContentResponse, Part};
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

pin_project! {
    /// Google AI stream parser.
    pub struct GoogleStreamParser<S> {
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
enum PartState {
    Text {
        content: String,
    },
    FunctionCall {
        #[allow(dead_code)]
        name: String,
        #[allow(dead_code)]
        args: String,
    },
    Thinking {
        content: String,
    },
}

impl<S> GoogleStreamParser<S>
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

impl<S> Stream for GoogleStreamParser<S>
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
            // Try to parse complete JSON objects from buffer
            // Google sends each chunk as a complete JSON object on its own line
            while let Some(line_end) = this.buffer.find('\n') {
                let line = this.buffer.drain(..=line_end).collect::<String>();
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                // Skip "data: " prefix if present
                let json_str = line.strip_prefix("data: ").unwrap_or(line);

                // Handle [DONE] marker
                if json_str == "[DONE]" {
                    *this.done = true;
                    return Poll::Ready(None);
                }

                // Parse the JSON response
                match serde_json::from_str::<GenerateContentResponse>(json_str) {
                    Ok(response) => {
                        if let Some(event) =
                            process_response(&response, this.parts, this.next_part_index, this.done)
                        {
                            return Poll::Ready(Some(event));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse Google stream chunk: {} - {}", e, json_str);
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
                        let json_str = remaining.trim();
                        let json_str = json_str.strip_prefix("data: ").unwrap_or(json_str);

                        if !json_str.is_empty() && json_str != "[DONE]" {
                            if let Ok(response) =
                                serde_json::from_str::<GenerateContentResponse>(json_str)
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
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Process a response chunk into stream events.
fn process_response(
    response: &GenerateContentResponse,
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

                let tool_part = ToolCallPart::new(&function_call.name, function_call.args.clone());

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
            Part::Thought { thought } => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    fn make_chunk(json: &str) -> Bytes {
        Bytes::from(format!("{}\n", json))
    }

    #[tokio::test]
    async fn test_parse_text_response() {
        let chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}"#;
        let bytes = vec![Ok(make_chunk(chunk))];
        let stream = stream::iter(bytes);
        let mut parser = GoogleStreamParser::new(stream);

        let event = parser.next().await.unwrap().unwrap();
        assert!(
            matches!(event, ModelResponseStreamEvent::PartStart(_)),
            "Expected PartStart, got {:?}",
            event
        );
    }

    #[tokio::test]
    async fn test_parse_function_call() {
        let chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"search","args":{"q":"rust"}}}]}}]}"#;
        let bytes = vec![Ok(make_chunk(chunk))];
        let stream = stream::iter(bytes);
        let mut parser = GoogleStreamParser::new(stream);

        let event = parser.next().await.unwrap().unwrap();
        if let ModelResponseStreamEvent::PartStart(start) = event {
            assert!(
                matches!(start.part, ModelResponsePart::ToolCall(_)),
                "Expected ToolCall"
            );
        } else {
            panic!("Expected PartStart");
        }
    }

    #[tokio::test]
    async fn test_parse_with_finish() {
        let chunk1 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}"#;
        let chunk2 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" World"}]},"finishReason":"STOP"}]}"#;
        let bytes = vec![Ok(make_chunk(chunk1)), Ok(make_chunk(chunk2))];
        let stream = stream::iter(bytes);
        let mut parser = GoogleStreamParser::new(stream);

        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        assert!(
            events.len() >= 2,
            "Expected at least 2 events, got {:?}",
            events
        );
    }
}
