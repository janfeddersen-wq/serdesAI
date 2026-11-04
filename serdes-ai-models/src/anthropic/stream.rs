//! Anthropic SSE stream parser.
//!
//! This module provides streaming support for Anthropic's Messages API.

use super::error::map_anthropic_error;
use super::types::{ContentBlockDelta, ContentBlockStart, StreamEvent};
use crate::error::ModelError;
use crate::refusal::{STOP_DETAILS_KEY, StopDetails, terminal_details};
use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;
use serde_json::Value as JsonValue;
use serdes_ai_core::ModelResponsePart;
use serdes_ai_core::messages::{
    FinishReason, ModelResponsePartDelta, ModelResponseStreamEvent, PartDeltaEvent, PartEndEvent,
    PartStartEvent, StreamCompleteEvent, TerminalMetadata, TextPart, ThinkingPart,
    ThinkingPartDelta, ToolCallPart,
};
use serdes_ai_streaming::{SseParser, StreamError};
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    /// Anthropic SSE stream parser.
    pub struct AnthropicStreamParser<S> {
        #[pin]
        inner: S,
        sse: SseParser,
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
        // Provider-reported stop reason from message_delta
        stop_reason: Option<String>,
        // Provider-reported stop details from message_delta (never the explanation)
        stop_details: Option<StopDetails>,
        // Raw input_transformations from message_start or message_delta
        input_transformations: Option<JsonValue>,
        // Whether message_stop was observed
        message_stop_seen: bool,
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
            sse: SseParser::new(),
            blocks: HashMap::new(),
            message_id: None,
            model: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            stop_reason: None,
            stop_details: None,
            input_transformations: None,
            message_stop_seen: false,
            done: false,
        }
    }

    /// The provider-reported stop reason (from `message_delta`), if any.
    ///
    /// This is only reliable after the stream has completed successfully
    /// (i.e. `message_stop` was observed).
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    /// The provider-reported stop details (from `message_delta`), if any.
    pub fn stop_details(&self) -> Option<&StopDetails> {
        self.stop_details.as_ref()
    }

    /// Whether `message_stop` was observed before the stream ended.
    pub fn message_stop_seen(&self) -> bool {
        self.message_stop_seen
    }

    /// Input tokens reported by the provider.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    /// Output tokens reported by the provider.
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }
}

impl<S> Stream for AnthropicStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // After StreamComplete has been emitted the stream is done.
        if *this.done {
            return Poll::Ready(None);
        }

        loop {
            while let Some(event) = this.sse.next_event() {
                if let Some(result) = process_event(
                    event.event.as_deref().unwrap_or("message"),
                    &event.data,
                    this.blocks,
                    this.message_id,
                    this.model,
                    this.input_tokens,
                    this.output_tokens,
                    this.cache_creation_tokens,
                    this.cache_read_tokens,
                    this.stop_reason,
                    this.stop_details,
                    this.input_transformations,
                    this.message_stop_seen,
                    this.done,
                ) {
                    if result.is_err()
                        || matches!(result, Ok(ModelResponseStreamEvent::StreamComplete(_)))
                    {
                        *this.done = true;
                    }
                    return Poll::Ready(Some(result));
                }
            }

            // If message_stop was already seen and all buffered events consumed,
            // we should not poll the inner stream further — just end.
            if *this.message_stop_seen {
                *this.done = true;
                return Poll::Ready(None);
            }

            // Need more data
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Err(error) = this.sse.feed(&bytes) {
                        *this.done = true;
                        return Poll::Ready(Some(Err(map_sse_error(error))));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    *this.done = true;
                    return Poll::Ready(Some(Err(ModelError::Other(e.into()))));
                }
                Poll::Ready(None) => {
                    *this.done = true;

                    if let Err(error) = this.sse.finish() {
                        return Poll::Ready(Some(Err(map_sse_error(error))));
                    }

                    if !*this.message_stop_seen {
                        return Poll::Ready(Some(Err(ModelError::incomplete_stream(
                            "stream ended before message_stop was received",
                        ))));
                    }

                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn map_sse_error(error: StreamError) -> ModelError {
    match error {
        StreamError::IncompleteSse => {
            ModelError::incomplete_stream("stream ended with an incomplete SSE record")
        }
        StreamError::InvalidUtf8 => ModelError::invalid_response("invalid UTF-8 in SSE stream"),
        StreamError::BufferOverflow => {
            ModelError::invalid_response("SSE record exceeded buffer limit")
        }
        other => ModelError::invalid_response(other.to_string()),
    }
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
    stop_reason: &mut Option<String>,
    stop_details: &mut Option<StopDetails>,
    input_transformations: &mut Option<JsonValue>,
    message_stop_seen: &mut bool,
    done: &mut bool,
) -> Option<Result<ModelResponseStreamEvent, ModelError>> {
    // Parse the JSON data — reject malformed events rather than silently dropping them
    let event: StreamEvent = match serde_json::from_str(data) {
        Ok(e) => e,
        Err(e) => {
            // stop_details may carry an explanation that must never reach a
            // log or an error message, so such data is not echoed.
            let message = if data.contains(STOP_DETAILS_KEY) {
                format!("Failed to parse stream event: {e} - data omitted (holds stop_details)")
            } else {
                format!("Failed to parse stream event: {} - data: {}", e, data)
            };
            return Some(Err(ModelError::invalid_response(message)));
        }
    };

    match event {
        StreamEvent::MessageStart { message } => {
            *message_id = Some(message.id);
            *model = Some(message.model);
            *input_tokens = message.usage.input_tokens;
            *cache_creation_tokens = message.usage.cache_creation_input_tokens;
            *cache_read_tokens = message.usage.cache_read_input_tokens;
            if message.input_transformations.is_some() {
                *input_transformations = message.input_transformations;
            }
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
                        // If the accumulated JSON is still just the initial empty
                        // object, replace it with the first real delta.
                        if input_json == "{}" {
                            input_json.clear();
                            input_json.push_str(&partial_json);
                        } else {
                            input_json.push_str(&partial_json);
                        }
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
            // Validate tool input JSON before accepting the block as complete.
            // Skip validation when the input is the default empty object (no
            // deltas were received).
            if let Some(BlockState::ToolUse { input_json, .. }) = blocks.get(&index) {
                if !input_json.is_empty()
                    && input_json != "{}"
                    && serde_json::from_str::<serde_json::Value>(input_json).is_err()
                {
                    return Some(Err(ModelError::invalid_response(format!(
                        "content_block_stop for tool at index {} has incomplete JSON input: {}",
                        index, input_json
                    ))));
                }
            }
            blocks.remove(&index);
            Some(Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent {
                index,
            })))
        }

        StreamEvent::MessageDelta {
            delta,
            usage,
            stop_details: outer_stop_details,
            input_transformations: outer_input_transformations,
        } => {
            if let Some(reason) = &delta.stop_reason {
                *stop_reason = Some(reason.clone());
            }
            // Accepted inside `delta` and beside it; inside wins.
            if let Some(details) = delta.stop_details.or(outer_stop_details) {
                *stop_details = Some(details);
            }
            if let Some(transformations) =
                delta.input_transformations.or(outer_input_transformations)
            {
                *input_transformations = Some(transformations);
            }
            if let Some(u) = usage {
                *output_tokens = u.output_tokens;
            }
            None
        }

        StreamEvent::MessageStop => {
            // Atomically validate terminal state before accepting message_stop.
            if !blocks.is_empty() {
                let open_indices: Vec<usize> = blocks.keys().copied().collect();
                *done = true;
                return Some(Err(ModelError::incomplete_stream(format!(
                    "message_stop received with open content block(s) at index/indices: {:?}",
                    open_indices
                ))));
            }

            *message_stop_seen = true;

            // Map the Anthropic stop reason to FinishReason
            let finish_reason = map_stop_reason(stop_reason.as_deref());

            // Emit StreamComplete with provider terminal metadata
            let mut event = StreamCompleteEvent::new(finish_reason)
                .with_input_tokens(*input_tokens)
                .with_output_tokens(*output_tokens);
            if let Some(tokens) = *cache_creation_tokens {
                event = event.with_cache_creation_tokens(tokens);
            }
            if let Some(tokens) = *cache_read_tokens {
                event = event.with_cache_read_tokens(tokens);
            }
            // Terminal metadata only when the provider sent stop details or
            // input transformations: a normal end_turn keeps it None.
            if let Some(details) = terminal_details(
                stop_reason.as_deref(),
                stop_details.as_ref(),
                input_transformations.as_ref(),
            ) {
                event.metadata = Some(TerminalMetadata {
                    response_id: message_id.clone(),
                    model: model.clone(),
                    details: Some(details),
                });
            }

            Some(Ok(ModelResponseStreamEvent::StreamComplete(event)))
        }

        StreamEvent::Ping => None,

        StreamEvent::Error { error } => Some(Err(map_anthropic_error(
            error.error_type,
            error.message,
            None,
            None,
        ))),
    }
}

/// Map an Anthropic stop reason string to a [`FinishReason`].
fn map_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") => FinishReason::EndTurn,
        Some("stop_sequence") => FinishReason::StopSequence,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCall,
        Some("refusal") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;

    fn make_sse_bytes(event_type: &str, data: &str) -> Bytes {
        Bytes::from(format!("event: {}\ndata: {}\n\n", event_type, data))
    }

    const OPUS_MSG_START: &str = r#"{"type":"message_start","message":{"id":"msg_r","type":"message","role":"assistant","model":"claude-opus-5-5","usage":{"input_tokens":10,"output_tokens":0}}}"#;

    /// A stream of `message_start`, the given middle events, then
    /// `message_delta` with `msg_delta` and `message_stop`; all events.
    async fn run_stream(
        middle: &[(&str, &str)],
        msg_delta: &str,
    ) -> Vec<Result<ModelResponseStreamEvent, ModelError>> {
        let mut bytes = vec![Ok(make_sse_bytes("message_start", OPUS_MSG_START))];
        for (event_type, data) in middle {
            bytes.push(Ok(make_sse_bytes(event_type, data)));
        }
        bytes.push(Ok(make_sse_bytes("message_delta", msg_delta)));
        bytes.push(Ok(make_sse_bytes(
            "message_stop",
            r#"{"type":"message_stop"}"#,
        )));
        AnthropicStreamParser::new(stream::iter(bytes))
            .collect()
            .await
    }

    fn terminal(events: &[Result<ModelResponseStreamEvent, ModelError>]) -> &StreamCompleteEvent {
        match events.last() {
            Some(Ok(ModelResponseStreamEvent::StreamComplete(complete))) => complete,
            other => panic!("expected StreamComplete last, got {other:?}"),
        }
    }

    const REFUSAL_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"reasoning_extraction","explanation":"private text"}},"usage":{"output_tokens":3}}"#;

    #[tokio::test]
    async fn refusal_maps_to_content_filter_with_category_in_metadata() {
        let events = run_stream(&[], REFUSAL_DELTA).await;

        let complete = terminal(&events);
        assert_eq!(complete.finish_reason, FinishReason::ContentFilter);
        let metadata = complete
            .metadata
            .as_ref()
            .expect("refusal carries metadata");
        assert_eq!(metadata.response_id.as_deref(), Some("msg_r"));
        let details = metadata.details.as_ref().unwrap();
        assert_eq!(details["stop_reason"], "refusal");
        assert_eq!(
            details["stop_details"],
            serde_json::json!({"type": "refusal", "category": "reasoning_extraction"})
        );
        assert!(!details.to_string().contains("private text"));
    }

    /// Opus 5.5 always thinks: the refusal usually follows a thinking block
    /// with an empty text (display "omitted") and a signature.
    #[tokio::test]
    async fn refusal_after_thinking_block_keeps_the_part_and_the_category() {
        let events = run_stream(
            &[
                (
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
                ),
                (
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#,
                ),
                (
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
            ],
            REFUSAL_DELTA,
        )
        .await;

        assert!(matches!(
            &events[0],
            Ok(ModelResponseStreamEvent::PartStart(start))
                if matches!(start.part, ModelResponsePart::Thinking(_))
        ));
        let complete = terminal(&events);
        assert_eq!(complete.finish_reason, FinishReason::ContentFilter);
        let details = complete
            .metadata
            .as_ref()
            .unwrap()
            .details
            .as_ref()
            .unwrap();
        assert_eq!(
            StopDetails::from_details(details)
                .unwrap()
                .category
                .as_deref(),
            Some("reasoning_extraction")
        );
    }

    #[tokio::test]
    async fn stop_details_beside_delta_are_read_too() {
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null},"stop_details":{"type":"refusal","category":"cyber"},"usage":{"output_tokens":1}}"#;

        let events = run_stream(&[], msg_delta).await;

        let complete = terminal(&events);
        assert_eq!(complete.finish_reason, FinishReason::ContentFilter);
        let details = complete
            .metadata
            .as_ref()
            .unwrap()
            .details
            .as_ref()
            .unwrap();
        assert_eq!(details["stop_details"]["category"], "cyber");
    }

    /// A stop_details shape the parser does not expect must neither fail the
    /// stream nor leak the explanation.
    #[tokio::test]
    async fn odd_stop_details_shape_still_parses() {
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"category":null,"type":7,"explanation":"private text"}},"usage":{"output_tokens":1}}"#;

        let events = run_stream(&[], msg_delta).await;

        let complete = terminal(&events);
        assert_eq!(complete.finish_reason, FinishReason::ContentFilter);
        let details = complete
            .metadata
            .as_ref()
            .unwrap()
            .details
            .as_ref()
            .unwrap();
        assert_eq!(details["stop_details"], serde_json::json!({}));
        assert!(!details.to_string().contains("private text"));
    }

    #[tokio::test]
    async fn parse_error_does_not_echo_stop_details() {
        // `usage.output_tokens` has the wrong type, so the event fails to parse.
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"explanation":"private text"}},"usage":{"output_tokens":"x"}}"#;

        let events = run_stream(&[], msg_delta).await;

        let error = events
            .iter()
            .find_map(|e| e.as_ref().err())
            .expect("parse error");
        let message = error.to_string();
        assert!(message.contains("Failed to parse stream event"));
        assert!(!message.contains("private text"));
    }

    #[tokio::test]
    async fn input_transformations_reach_the_metadata() {
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"input_transformations":[{"type":"thinking_block_dropped","index":2}]},"usage":{"output_tokens":1}}"#;

        let events = run_stream(&[], msg_delta).await;

        let complete = terminal(&events);
        assert_eq!(complete.finish_reason, FinishReason::EndTurn);
        let details = complete
            .metadata
            .as_ref()
            .unwrap()
            .details
            .as_ref()
            .unwrap();
        assert_eq!(
            details["input_transformations"],
            serde_json::json!([{"type": "thinking_block_dropped", "index": 2}])
        );
        assert!(details.get("stop_details").is_none());
    }

    #[tokio::test]
    async fn normal_end_turn_keeps_metadata_none() {
        let msg_delta = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#;

        let events = run_stream(&[], msg_delta).await;

        let complete = terminal(&events);
        assert_eq!(complete.finish_reason, FinishReason::EndTurn);
        assert!(complete.metadata.is_none());
    }

    #[tokio::test]
    async fn test_parse_message_start() {
        let data = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;
        let bytes = vec![
            Ok(make_sse_bytes("message_start", data)),
            Ok(make_sse_bytes("message_stop", msg_stop)),
        ];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // Message start emits nothing; message_stop emits StreamComplete
        let event = parser.next().await;
        assert!(event.is_some());
        assert!(
            matches!(
                event.unwrap().unwrap(),
                ModelResponseStreamEvent::StreamComplete(_)
            ),
            "expected StreamComplete"
        );
        // Next poll should be None (done)
        assert!(parser.next().await.is_none());
    }

    #[tokio::test]
    async fn test_parse_text_stream() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
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

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // Collect events
        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Should have: PartStart, PartDelta, PartEnd, StreamComplete
        assert_eq!(events.len(), 4, "Expected 4 events, got {:?}", events);

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
        assert!(
            matches!(&events[3], ModelResponseStreamEvent::StreamComplete(_)),
            "Fourth should be StreamComplete"
        );

        // Provider terminal metadata should survive
        assert_eq!(parser.stop_reason(), Some("end_turn"));
        assert!(parser.message_stop_seen());
        assert_eq!(parser.output_tokens(), 5);
    }

    #[tokio::test]
    async fn test_parse_tool_use_stream() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"search","input":{}}}"#;
        let delta1 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#;
        let delta2 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta1)),
            Ok(make_sse_bytes("content_block_delta", delta2)),
            Ok(make_sse_bytes("content_block_stop", block_stop)),
            Ok(make_sse_bytes("message_stop", msg_stop)),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            events.push(result.unwrap());
        }

        // Should have: PartStart, PartDelta, PartDelta, PartEnd, StreamComplete
        assert_eq!(events.len(), 5, "Expected 5 events, got {:?}", events);

        // First should be PartStart with tool_use
        if let ModelResponseStreamEvent::PartStart(start) = &events[0] {
            assert!(
                matches!(&start.part, ModelResponsePart::ToolCall(_)),
                "Expected ToolCall"
            );
        } else {
            panic!("Expected PartStart");
        }
        // Last should be StreamComplete
        assert!(
            matches!(&events[4], ModelResponseStreamEvent::StreamComplete(_)),
            "Last should be StreamComplete"
        );
    }

    async fn parse_stream_error(code: &str, message: &str) -> ModelError {
        let error =
            format!(r#"{{"type":"error","error":{{"type":"{code}","message":"{message}"}}}}"#);
        let stream = stream::iter(vec![Ok(make_sse_bytes("error", &error))]);
        let mut parser = AnthropicStreamParser::new(stream);
        parser.next().await.unwrap().unwrap_err()
    }

    #[tokio::test]
    async fn test_parse_rate_limit_error() {
        let err = parse_stream_error("rate_limit_error", "Rate limited").await;

        assert!(err.is_rate_limited());
        assert!(err.is_retryable());
        match err {
            ModelError::Provider {
                code,
                message,
                kind,
                ..
            } => {
                assert_eq!(code, "rate_limit_error");
                assert_eq!(message, "Rate limited");
                assert_eq!(kind, crate::ProviderErrorKind::RateLimited);
            }
            other => panic!("expected provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_overloaded_error() {
        let err = parse_stream_error("overloaded_error", "Overloaded").await;
        assert!(err.is_transient());
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn test_parse_permanent_errors() {
        for code in ["authentication_error", "invalid_request_error"] {
            let err = parse_stream_error(code, "Permanent").await;
            assert!(!err.is_retryable(), "{code} must not be retryable");
        }
    }

    #[tokio::test]
    async fn test_parse_ping() {
        let ping = r#"{"type":"ping"}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;
        let bytes = vec![
            Ok(make_sse_bytes("ping", ping)),
            Ok(make_sse_bytes("message_stop", msg_stop)),
        ];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // Ping emits nothing; message_stop emits StreamComplete
        let event = parser.next().await;
        assert!(event.is_some());
        assert!(
            matches!(
                event.unwrap().unwrap(),
                ModelResponseStreamEvent::StreamComplete(_)
            ),
            "expected StreamComplete"
        );
        // Done
        assert!(parser.next().await.is_none());
    }

    // ========================================================================
    // Regression tests for issue #39: premature EOF must not be treated as
    // successful completion.
    // ========================================================================

    /// Helper: build a partial (incomplete) SSE frame as bytes.
    fn make_partial_sse_bytes(event_type: &str, partial_data: &str) -> Bytes {
        // Intentionally missing the trailing blank line so the frame is incomplete
        Bytes::from(format!(
            "event: {}
data: {}",
            event_type, partial_data
        ))
    }

    /// Test 1: EOF before any `message_stop`.
    #[tokio::test]
    async fn test_eof_before_message_stop() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta)),
            Ok(make_sse_bytes("content_block_stop", block_stop)),
            // NO message_delta, NO message_stop — stream just ends
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut had_error = false;
        while let Some(result) = parser.next().await {
            if let Err(ref e) = result {
                assert!(
                    matches!(e, ModelError::IncompleteStream(_)),
                    "expected IncompleteStream, got: {:?}",
                    e
                );
                had_error = true;
            }
        }
        assert!(had_error, "expected an IncompleteStream error");
        assert!(!parser.message_stop_seen());
    }

    /// Test 2: EOF with a partial final SSE record in the buffer.
    #[tokio::test]
    async fn test_eof_with_partial_sse_frame() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta)),
            Ok(make_sse_bytes("content_block_stop", block_stop)),
            // A partial frame that will never be completed — inner stream ends
            Ok(make_partial_sse_bytes(
                "message_delta",
                r#"{"type":"message_delta","#,
            )),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut had_error = false;
        while let Some(result) = parser.next().await {
            if let Err(ref e) = result {
                assert!(
                    matches!(e, ModelError::IncompleteStream(_)),
                    "expected IncompleteStream for partial frame, got: {:?}",
                    e
                );
                had_error = true;
            }
        }
        assert!(
            had_error,
            "expected IncompleteStream error for partial SSE frame"
        );
    }

    /// Test 3: EOF with an open text block.
    #[tokio::test]
    async fn test_eof_with_open_text_block() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        // NO content_block_stop, NO message_stop

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta)),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut had_error = false;
        while let Some(result) = parser.next().await {
            if let Err(ref e) = result {
                assert!(
                    matches!(e, ModelError::IncompleteStream(_)),
                    "expected IncompleteStream for open text block, got: {:?}",
                    e
                );
                had_error = true;
            }
        }
        assert!(
            had_error,
            "expected IncompleteStream error for open text block"
        );
    }

    /// Test 4: EOF with an open thinking block.
    #[tokio::test]
    async fn test_eof_with_open_thinking_block() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hmm"}}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta)),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut had_error = false;
        while let Some(result) = parser.next().await {
            if let Err(ref e) = result {
                assert!(
                    matches!(e, ModelError::IncompleteStream(_)),
                    "expected IncompleteStream for open thinking block, got: {:?}",
                    e
                );
                had_error = true;
            }
        }
        assert!(
            had_error,
            "expected IncompleteStream error for open thinking block"
        );
    }

    /// Test 5: EOF with an open tool-use block (incomplete tool JSON).
    #[tokio::test]
    async fn test_eof_with_open_tool_use_block() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool_1","name":"search","input":{}}}"#;
        let delta1 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#;
        // Second delta makes the JSON incomplete (missing closing brace)
        let delta2 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"rust\""}}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta1)),
            Ok(make_sse_bytes("content_block_delta", delta2)),
            // NO content_block_stop, NO message_stop
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut had_error = false;
        while let Some(result) = parser.next().await {
            if let Err(ref e) = result {
                assert!(
                    matches!(e, ModelError::IncompleteStream(_)),
                    "expected IncompleteStream for open tool block, got: {:?}",
                    e
                );
                had_error = true;
            }
        }
        assert!(
            had_error,
            "expected IncompleteStream error for open tool-use block"
        );
    }

    /// Test 6: `content_block_stop` followed by EOF without `message_stop`.
    #[tokio::test]
    async fn test_content_block_stop_then_eof_without_message_stop() {
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
            // block is closed but no message_stop
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut had_error = false;
        while let Some(result) = parser.next().await {
            if let Err(ref e) = result {
                assert!(
                    matches!(e, ModelError::IncompleteStream(_)),
                    "expected IncompleteStream without message_stop, got: {:?}",
                    e
                );
                had_error = true;
            }
        }
        assert!(had_error, "expected IncompleteStream error");
        assert!(!parser.message_stop_seen());
    }

    /// Test 7: A valid sequence ending in `message_stop` produces no error.
    #[tokio::test]
    async fn test_valid_stream_with_message_stop() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":7}}}"#;
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

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut saw_stream_complete = false;
        while let Some(result) = parser.next().await {
            match result {
                Ok(ModelResponseStreamEvent::StreamComplete(sc)) => {
                    saw_stream_complete = true;
                    assert_eq!(sc.finish_reason, FinishReason::EndTurn);
                    assert_eq!(sc.input_tokens, Some(10));
                    assert_eq!(sc.output_tokens, Some(5));
                    assert_eq!(sc.cache_creation_tokens, Some(3));
                    assert_eq!(sc.cache_read_tokens, Some(7));
                }
                Ok(_) => {}
                Err(e) => panic!("valid stream should not produce error: {:?}", e),
            }
        }
        assert!(
            saw_stream_complete,
            "valid stream should emit StreamComplete"
        );
        assert!(parser.message_stop_seen());
        assert_eq!(parser.stop_reason(), Some("end_turn"));
        assert_eq!(parser.output_tokens(), 5);
        assert_eq!(parser.input_tokens(), 10);
    }

    /// Test 8: Premature EOF after visible text produces an error and no
    /// further successful events.
    #[tokio::test]
    async fn test_premature_eof_after_text_produces_error() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta1 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let delta2 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta1)),
            Ok(make_sse_bytes("content_block_delta", delta2)),
            // Stream ends here — open text block, no message_stop
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let mut text_deltas_seen = 0;
        let mut error_seen = false;
        while let Some(result) = parser.next().await {
            match result {
                Ok(ModelResponseStreamEvent::PartDelta(PartDeltaEvent {
                    delta: ModelResponsePartDelta::Text(_),
                    ..
                })) => {
                    text_deltas_seen += 1;
                }
                Ok(_) => {}
                Err(e) => {
                    assert!(
                        matches!(e, ModelError::IncompleteStream(_)),
                        "expected IncompleteStream, got: {:?}",
                        e
                    );
                    error_seen = true;
                }
            }
        }
        assert!(text_deltas_seen >= 2, "should have seen text deltas");
        assert!(error_seen, "must see IncompleteStream error");
        assert!(!parser.message_stop_seen());
    }

    // =====================================================
    // Issue #40 regression tests
    // =====================================================

    /// Helper: create a stream that feeds bytes one chunk at a time.
    fn make_byte_chunks(full: &[u8], chunk_size: usize) -> Vec<Result<Bytes, reqwest::Error>> {
        full.chunks(chunk_size)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect()
    }

    /// Full valid stream as raw bytes (message_start through message_stop).
    fn full_valid_stream_bytes() -> Vec<u8> {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&make_sse_bytes("message_start", msg_start));
        bytes.extend_from_slice(&make_sse_bytes("content_block_start", block_start));
        bytes.extend_from_slice(&make_sse_bytes("content_block_delta", delta));
        bytes.extend_from_slice(&make_sse_bytes("content_block_stop", block_stop));
        bytes.extend_from_slice(&make_sse_bytes("message_delta", r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#));
        bytes.extend_from_slice(&make_sse_bytes("message_stop", msg_stop));
        bytes
    }

    /// Collect all events from a parser, returning Ok(events) or Err(error).
    async fn collect_events<S>(
        mut parser: AnthropicStreamParser<S>,
    ) -> Result<Vec<ModelResponseStreamEvent>, ModelError>
    where
        S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    {
        let mut events = Vec::new();
        while let Some(result) = parser.next().await {
            match result {
                Ok(e) => events.push(e),
                Err(e) => return Err(e),
            }
        }
        Ok(events)
    }

    #[tokio::test]
    async fn test_byte_fragmentation_all_boundaries() {
        let full = full_valid_stream_bytes();

        for chunk_size in 1..=full.len() {
            let chunks = make_byte_chunks(&full, chunk_size);
            let stream = stream::iter(chunks);
            let parser = AnthropicStreamParser::new(stream);

            let events = collect_events(parser).await.unwrap_or_else(|e| {
                panic!("chunk_size {}: stream failed with error: {}", chunk_size, e);
            });

            assert_eq!(
                events.len(),
                4,
                "chunk_size {}: expected 4 events (PartStart, PartDelta, PartEnd), got {}",
                chunk_size,
                events.len()
            );
        }
    }

    #[tokio::test]
    async fn test_multibyte_utf8_split_at_every_boundary() {
        // Use a multibyte character (U+00E9 = C3 A9) in the text delta
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"caf\u00e9"}}"#;
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;

        let full = {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&make_sse_bytes("message_start", msg_start));
            bytes.extend_from_slice(&make_sse_bytes("content_block_start", block_start));
            bytes.extend_from_slice(&make_sse_bytes("content_block_delta", delta));
            bytes.extend_from_slice(&make_sse_bytes("content_block_stop", block_stop));
            bytes.extend_from_slice(&make_sse_bytes("message_stop", msg_stop));
            bytes
        };

        // Split at every byte boundary — must reconstruct perfectly
        for split_point in 1..full.len() {
            let chunks = vec![
                Ok(Bytes::copy_from_slice(&full[..split_point])),
                Ok(Bytes::copy_from_slice(&full[split_point..])),
            ];
            let stream = stream::iter(chunks);
            let parser = AnthropicStreamParser::new(stream);

            let events = collect_events(parser).await.unwrap_or_else(|e| {
                panic!(
                    "split_point {}: stream failed with error: {}",
                    split_point, e
                );
            });

            assert_eq!(
                events.len(),
                4,
                "split_point {}: expected 4 events",
                split_point
            );
        }
    }

    #[tokio::test]
    async fn test_malformed_json_yields_typed_error() {
        let bad_data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"broken"#;
        let bytes = vec![Ok(make_sse_bytes("content_block_delta", bad_data))];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let event = parser.next().await.unwrap();
        assert!(event.is_err(), "expected error for malformed JSON");
        let err = event.unwrap_err();
        assert!(
            err.to_string().contains("Failed to parse stream event"),
            "error should mention parse failure, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_irrecoverably_invalid_utf8_yields_error() {
        // Build a valid SSE frame but inject invalid UTF-8 byte (0xFF)
        let valid = make_sse_bytes("ping", r#"{"type":"ping"}"#);
        let mut bad_bytes = valid.to_vec();
        // Insert an invalid UTF-8 byte in the data portion
        bad_bytes.insert(20, 0xFF);

        let bytes = vec![Ok(Bytes::from(bad_bytes))];
        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        let event = parser.next().await;
        assert!(event.is_some(), "expected an error event for invalid UTF-8");
        if let Some(Err(err)) = event {
            assert!(
                err.to_string().contains("invalid UTF-8"),
                "error should mention invalid UTF-8, got: {}",
                err
            );
        } else {
            panic!("expected Err, got Ok or None");
        }
    }

    #[tokio::test]
    async fn test_malformed_record_between_valid_deltas_no_success() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let delta1 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let bad_data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"broken"#;
        let delta2 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"World"}}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_start", block_start)),
            Ok(make_sse_bytes("content_block_delta", delta1)),
            Ok(make_sse_bytes("content_block_delta", bad_data)),
            Ok(make_sse_bytes("content_block_delta", delta2)),
        ];

        let stream = stream::iter(bytes);
        let parser = AnthropicStreamParser::new(stream);

        let result = collect_events(parser).await;

        assert!(
            result.is_err(),
            "stream with malformed JSON between valid deltas must produce an error"
        );

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to parse stream event"),
            "error should mention parse failure, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_no_events_after_fatal_parse_error() {
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let bad_data = r#"not valid json at all"#;
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"should-not-see"}}"#;

        let bytes = vec![
            Ok(make_sse_bytes("message_start", msg_start)),
            Ok(make_sse_bytes("content_block_delta", bad_data)),
            Ok(make_sse_bytes("content_block_delta", delta)),
        ];

        let stream = stream::iter(bytes);
        let mut parser = AnthropicStreamParser::new(stream);

        // First poll should return the error
        let event = parser.next().await;
        assert!(event.is_some(), "expected error event");
        assert!(
            event.as_ref().unwrap().is_err(),
            "expected Err for malformed JSON"
        );

        // Subsequent polls should return None (parser is done)
        let after = parser.next().await;
        assert!(
            after.is_none(),
            "no events should be emitted after fatal parse error"
        );
    }

    #[tokio::test]
    async fn test_fragmentation_preserves_event_sequence() {
        // Verify that byte-level fragmentation doesn't change the event sequence
        let full = full_valid_stream_bytes();

        // Single chunk (baseline)
        let single = make_byte_chunks(&full, full.len());
        let single_events = collect_events(AnthropicStreamParser::new(stream::iter(single)))
            .await
            .unwrap();

        // Fragmented into 1-byte chunks (worst case)
        let fragmented = make_byte_chunks(&full, 1);
        let frag_events = collect_events(AnthropicStreamParser::new(stream::iter(fragmented)))
            .await
            .unwrap();

        assert_eq!(
            single_events.len(),
            frag_events.len(),
            "fragmented stream should produce same number of events"
        );

        for (i, (s, f)) in single_events.iter().zip(frag_events.iter()).enumerate() {
            assert_eq!(
                std::mem::discriminant(s),
                std::mem::discriminant(f),
                "event {} type mismatch: single={:?} fragmented={:?}",
                i,
                s,
                f
            );
        }
    }

    #[tokio::test]
    async fn test_invalid_utf8_byte_split_across_chunks() {
        // A 3-byte UTF-8 character (U+4E9C = E4 BA 9C) split so byte 1 is in
        // chunk 1 and bytes 2-3 are in chunk 2.
        let msg_start = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        // The delta contains the multibyte char embedded in JSON
        let delta_json = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\u{4E9C}\"}}";
        let block_stop = r#"{"type":"content_block_stop","index":0}"#;
        let msg_stop = r#"{"type":"message_stop"}"#;

        let full = {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&make_sse_bytes("message_start", msg_start));
            bytes.extend_from_slice(&make_sse_bytes("content_block_start", block_start));
            bytes.extend_from_slice(&make_sse_bytes("content_block_delta", delta_json));
            bytes.extend_from_slice(&make_sse_bytes("content_block_stop", block_stop));
            bytes.extend_from_slice(&make_sse_bytes("message_stop", msg_stop));
            bytes
        };

        // Find the byte position of the multibyte char in the full stream
        let char_bytes = "\u{4E9C}".as_bytes(); // E4 BA 9C
        let char_pos = full
            .windows(char_bytes.len())
            .position(|w| w == char_bytes)
            .expect("multibyte char should be in stream");

        // Split right in the middle of the multibyte char
        for split_offset in 1..char_bytes.len() {
            let split_point = char_pos + split_offset;
            let chunks = vec![
                Ok(Bytes::copy_from_slice(&full[..split_point])),
                Ok(Bytes::copy_from_slice(&full[split_point..])),
            ];
            let stream = stream::iter(chunks);
            let parser = AnthropicStreamParser::new(stream);

            let events = collect_events(parser).await.unwrap_or_else(|e| {
                panic!("split_offset {}: failed with error: {}", split_offset, e);
            });

            assert_eq!(
                events.len(),
                4,
                "split_offset {}: expected 4 events",
                split_offset
            );
        }
    }
}
