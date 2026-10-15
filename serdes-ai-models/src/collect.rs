//! Collect a [`StreamedResponse`] into a complete [`ModelResponse`].

use crate::error::ModelError;
use crate::model::StreamedResponse;
use futures::StreamExt;
use serdes_ai_core::{FinishReason, ModelResponse, ModelResponseStreamEvent};

/// Drain a [`StreamedResponse`] into the [`ModelResponse`] it describes.
///
/// Parts are assembled exactly as streamed: a `PartStart` carries the part
/// (including provider identity such as ids, provider name and details) and
/// every `PartDelta` is applied to it with the delta logic of
/// `serdes_ai_core`. The terminal `StreamComplete` event supplies the finish
/// reason, usage and provider metadata (response id, model name, vendor
/// details). `model_name` stays `None` unless the provider metadata names it.
///
/// A stream that ends cleanly without a terminal event yields its parts with
/// finish reason [`FinishReason::Stop`] - several providers never emit one.
///
/// # Errors
///
/// - the first error the stream yields, unchanged;
/// - [`ModelError::InvalidResponse`] when a part event is inconsistent (a
///   start that skips an index, or a delta for a part that was never started
///   or is of another kind) - streamed content is never dropped silently;
/// - [`ModelError::IncompleteStream`] when the stream ends without any part
///   and without a terminal event.
pub async fn collect_stream(mut stream: StreamedResponse) -> Result<ModelResponse, ModelError> {
    let mut response = ModelResponse::new();
    let mut completed = false;

    while let Some(event) = stream.next().await {
        match event? {
            ModelResponseStreamEvent::PartStart(start) => {
                if start.index < response.parts.len() {
                    response.parts[start.index] = start.part;
                } else if start.index == response.parts.len() {
                    response.parts.push(start.part);
                } else {
                    return Err(ModelError::invalid_response(
                        "Stream part started out of order",
                    ));
                }
            }
            ModelResponseStreamEvent::PartDelta(delta) => {
                let applied = response
                    .parts
                    .get_mut(delta.index)
                    .is_some_and(|part| delta.delta.apply(part));
                if !applied {
                    return Err(ModelError::invalid_response(
                        "Stream delta does not match a started part",
                    ));
                }
            }
            ModelResponseStreamEvent::PartEnd(_) => {}
            ModelResponseStreamEvent::StreamComplete(complete) => {
                response.finish_reason = Some(complete.finish_reason);
                response.usage = complete.request_usage();
                if let Some(metadata) = &complete.metadata {
                    metadata.apply(&mut response);
                }
                completed = true;
            }
        }
    }

    if !completed {
        if response.parts.is_empty() {
            return Err(ModelError::incomplete_stream(
                "Model stream ended without producing any content",
            ));
        }
        response.finish_reason = Some(FinishReason::Stop);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_core::messages::{
        ModelResponsePartDelta, PartDeltaEvent, StreamCompleteEvent, TerminalMetadata,
        ThinkingPartDelta, ToolCallArgs,
    };
    use serdes_ai_core::{ModelResponsePart, TextPart, ThinkingPart, ToolCallPart};

    fn stream_of(events: Vec<Result<ModelResponseStreamEvent, ModelError>>) -> StreamedResponse {
        Box::pin(futures::stream::iter(events))
    }

    fn thinking_delta(
        index: usize,
        content: &str,
        signature: Option<&str>,
    ) -> ModelResponseStreamEvent {
        let mut delta = ThinkingPartDelta::new(content);
        delta.signature_delta = signature.map(str::to_owned);
        ModelResponseStreamEvent::PartDelta(PartDeltaEvent {
            index,
            delta: ModelResponsePartDelta::Thinking(delta),
        })
    }

    /// Text, thinking and tool-call parts are assembled from their start and
    /// delta events, keeping provider identity (id, provider name, signature);
    /// the terminal event supplies finish reason, usage and metadata.
    #[tokio::test]
    async fn collect_stream_assembles_text_thinking_and_tool_call_parts() {
        let mut complete = StreamCompleteEvent::new(FinishReason::ToolCall);
        complete.input_tokens = Some(11);
        complete.output_tokens = Some(5);
        complete.metadata = Some(TerminalMetadata {
            response_id: Some("resp_1".into()),
            model: Some("model-x".into()),
            details: Some(serde_json::json!({"provider": "test"})),
        });
        let events = vec![
            Ok(ModelResponseStreamEvent::part_start(
                0,
                ModelResponsePart::Thinking(
                    ThinkingPart::new("")
                        .with_id("rs_1")
                        .with_provider_name("openai"),
                ),
            )),
            Ok(thinking_delta(0, "Let me ", None)),
            Ok(thinking_delta(0, "check.", Some("opaque-signature"))),
            Ok(ModelResponseStreamEvent::part_end(0)),
            Ok(ModelResponseStreamEvent::part_start(
                1,
                ModelResponsePart::Text(TextPart::new("")),
            )),
            Ok(ModelResponseStreamEvent::text_delta(1, "Hel")),
            Ok(ModelResponseStreamEvent::text_delta(1, "lo")),
            Ok(ModelResponseStreamEvent::part_end(1)),
            Ok(ModelResponseStreamEvent::part_start(
                2,
                ModelResponsePart::ToolCall(
                    ToolCallPart::new("get_weather", ToolCallArgs::String(String::new()))
                        .with_tool_call_id("call_1"),
                ),
            )),
            Ok(ModelResponseStreamEvent::tool_call_delta(2, "{\"city\":")),
            Ok(ModelResponseStreamEvent::tool_call_delta(2, "\"Berlin\"}")),
            Ok(ModelResponseStreamEvent::part_end(2)),
            Ok(ModelResponseStreamEvent::StreamComplete(complete)),
        ];

        let response = collect_stream(stream_of(events))
            .await
            .expect("a complete stream collects");

        assert_eq!(response.parts.len(), 3);
        match &response.parts[0] {
            ModelResponsePart::Thinking(thinking) => {
                assert_eq!(thinking.content, "Let me check.");
                assert_eq!(thinking.id.as_deref(), Some("rs_1"));
                assert_eq!(thinking.provider_name.as_deref(), Some("openai"));
                assert_eq!(thinking.signature.as_deref(), Some("opaque-signature"));
            }
            other => panic!("expected a thinking part, got {other:?}"),
        }
        match &response.parts[1] {
            ModelResponsePart::Text(text) => assert_eq!(text.content, "Hello"),
            other => panic!("expected a text part, got {other:?}"),
        }
        match &response.parts[2] {
            ModelResponsePart::ToolCall(call) => {
                assert_eq!(call.tool_name, "get_weather");
                assert_eq!(call.tool_call_id.as_deref(), Some("call_1"));
                assert_eq!(call.args.to_json(), serde_json::json!({"city": "Berlin"}));
            }
            other => panic!("expected a tool-call part, got {other:?}"),
        }
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCall));
        let usage = response.usage.expect("usage from the terminal event");
        assert_eq!(usage.request_tokens, Some(11));
        assert_eq!(usage.response_tokens, Some(5));
        assert_eq!(response.vendor_id.as_deref(), Some("resp_1"));
        assert_eq!(response.model_name.as_deref(), Some("model-x"));
        assert_eq!(
            response.vendor_details,
            Some(serde_json::json!({"provider": "test"}))
        );
    }

    /// A stream that ends cleanly without a terminal event still yields its
    /// parts; the finish reason defaults to `Stop` (several providers never
    /// emit `StreamComplete`).
    #[tokio::test]
    async fn collect_stream_without_terminal_event_defaults_to_stop() {
        let events = vec![
            Ok(ModelResponseStreamEvent::part_start(
                0,
                ModelResponsePart::Text(TextPart::new("Hi")),
            )),
            Ok(ModelResponseStreamEvent::part_end(0)),
        ];

        let response = collect_stream(stream_of(events))
            .await
            .expect("parts without terminal event collect");

        assert_eq!(response.parts.len(), 1);
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert!(response.usage.is_none());
    }

    /// A stream that produced nothing at all is an incomplete stream, not an
    /// empty success.
    #[tokio::test]
    async fn collect_stream_empty_stream_is_incomplete() {
        let error = collect_stream(stream_of(Vec::new()))
            .await
            .expect_err("an empty stream must not collect");

        assert!(
            matches!(error, ModelError::IncompleteStream(_)),
            "expected an incomplete-stream error, got {error:?}"
        );
    }

    /// The first stream error ends collection and is returned unchanged.
    #[tokio::test]
    async fn collect_stream_propagates_stream_error() {
        let events = vec![
            Ok(ModelResponseStreamEvent::part_start(
                0,
                ModelResponsePart::Text(TextPart::new("partial")),
            )),
            Err(ModelError::invalid_response("broken frame")),
            Ok(ModelResponseStreamEvent::text_delta(0, "never applied")),
        ];

        let error = collect_stream(stream_of(events))
            .await
            .expect_err("a stream error must fail collection");

        assert!(error.to_string().contains("broken frame"), "got {error}");
    }

    /// Streamed content must never be dropped silently: a delta for a part
    /// that was never started, a delta of the wrong kind, and a start that
    /// skips an index are protocol violations.
    #[tokio::test]
    async fn collect_stream_rejects_inconsistent_part_events() {
        let unknown_part = vec![Ok(ModelResponseStreamEvent::text_delta(0, "orphan"))];
        let wrong_kind = vec![
            Ok(ModelResponseStreamEvent::part_start(
                0,
                ModelResponsePart::Text(TextPart::new("")),
            )),
            Ok(ModelResponseStreamEvent::tool_call_delta(0, "{}")),
        ];
        let skipped_index = vec![Ok(ModelResponseStreamEvent::part_start(
            1,
            ModelResponsePart::Text(TextPart::new("")),
        ))];

        for events in [unknown_part, wrong_kind, skipped_index] {
            let error = collect_stream(stream_of(events))
                .await
                .expect_err("an inconsistent part event must fail collection");
            assert!(
                matches!(error, ModelError::InvalidResponse(_)),
                "expected an invalid-response error, got {error:?}"
            );
        }
    }
}
