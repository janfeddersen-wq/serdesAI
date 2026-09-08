//! Mapping of wire stream events onto serdesAI model stream events.
//!
//! The wire protocol is verbose (lifecycle events for every part); serdesAI
//! only cares about part starts, deltas, part ends, and exactly one terminal
//! `StreamComplete` carrying the finish reason and usage. Terminal integrity
//! follows the crate-wide contract: the terminal event is always last, and a
//! failed turn surfaces as an error instead of a synthetic completion.

use super::wire::{OutputContent, OutputItem, ResponseObject, SummaryTextItem, codes};
use crate::ModelError;
use serde::{Deserialize, Serialize};
use serdes_ai_core::FinishReason;
use serdes_ai_core::ModelFailureKind;
use serdes_ai_core::messages::{
    ModelResponsePart, ModelResponsePartDelta, ModelResponseStreamEvent, PartDeltaEvent,
    PartEndEvent, PartStartEvent, StreamCompleteEvent, TextPart, TextPartDelta, ThinkingPart,
    ThinkingPartDelta, ToolCallArgs, ToolCallPart, ToolCallPartDelta,
};

/// A decoded data frame shared by the HTTP and websocket readers.
pub(super) enum DataEvent {
    Event(Box<StreamEvent>),
    Error(super::wire::WsErrorEnvelope),
}

pub(super) fn decode(payload: &str) -> Result<DataEvent, ModelError> {
    let value: serde_json::Value =
        serde_json::from_str(payload).map_err(|e| ModelError::invalid_response(e.to_string()))?;
    if value.get("type").and_then(serde_json::Value::as_str) == Some("error") {
        serde_json::from_value(value).map(DataEvent::Error)
    } else {
        serde_json::from_value(value).map(DataEvent::Event)
    }
    .map_err(|e| ModelError::invalid_response(e.to_string()))
}

/// A streaming event, used identically by SSE and the websocket transport.
///
/// Supported events carry a `sequence_number` within a single response.
/// Unknown external event names are ignored; malformed known events error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The response was created.
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// Sequence number.
        sequence_number: u64,
        /// The response object.
        response: ResponseObject,
    },
    /// The response is running.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        /// Sequence number.
        sequence_number: u64,
        /// The response object.
        response: ResponseObject,
    },
    /// An output item started.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        /// Sequence number.
        sequence_number: u64,
        /// Index of the item in `output`.
        output_index: u64,
        /// The item.
        item: OutputItem,
    },
    /// An output item finished.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        /// Sequence number.
        sequence_number: u64,
        /// Index of the item in `output`.
        output_index: u64,
        /// The item.
        item: OutputItem,
    },
    /// A content part was added to a message item.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the part in the item's content.
        content_index: u64,
        /// The content part.
        part: OutputContent,
    },
    /// A content part finished.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the part in the item's content.
        content_index: u64,
        /// The content part.
        part: OutputContent,
    },
    /// Text delta.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the part in the item's content.
        content_index: u64,
        /// The delta text.
        delta: String,
    },
    /// Text finished.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the part in the item's content.
        content_index: u64,
        /// The full text.
        text: String,
    },
    /// A reasoning summary part started.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the summary part.
        summary_index: u64,
        /// The summary part.
        part: SummaryTextItem,
    },
    /// A reasoning summary part finished.
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the summary part.
        summary_index: u64,
        /// The summary part.
        part: SummaryTextItem,
    },
    /// Reasoning summary delta.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the summary part.
        summary_index: u64,
        /// The delta text.
        delta: String,
    },
    /// Reasoning summary finished.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// Index of the summary part.
        summary_index: u64,
        /// The full summary text.
        text: String,
    },
    /// Function call arguments delta.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// The delta (a fragment of the JSON arguments string).
        delta: String,
    },
    /// Function call arguments finished.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        /// Sequence number.
        sequence_number: u64,
        /// Owning item ID.
        item_id: String,
        /// Index of the item in `output`.
        output_index: u64,
        /// The full JSON arguments string.
        arguments: String,
    },
    /// The response completed successfully.
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        /// Sequence number.
        sequence_number: u64,
        /// The final response object.
        response: ResponseObject,
    },
    /// The response failed.
    #[serde(rename = "response.failed")]
    ResponseFailed {
        /// Sequence number.
        sequence_number: u64,
        /// The final response object carrying `error`.
        response: ResponseObject,
    },
    /// The response stopped before completion.
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        /// Sequence number.
        sequence_number: u64,
        /// The final response object carrying `incomplete_details`.
        response: ResponseObject,
    },
    /// An external event outside the supported subset. It produces no model event.
    #[serde(other)]
    Unknown,
}

impl StreamEvent {
    /// The event's sequence number, or zero for an ignored unknown event.
    #[must_use]
    pub fn sequence_number(&self) -> u64 {
        match self {
            Self::ResponseCreated {
                sequence_number, ..
            }
            | Self::ResponseInProgress {
                sequence_number, ..
            }
            | Self::OutputItemAdded {
                sequence_number, ..
            }
            | Self::OutputItemDone {
                sequence_number, ..
            }
            | Self::ContentPartAdded {
                sequence_number, ..
            }
            | Self::ContentPartDone {
                sequence_number, ..
            }
            | Self::OutputTextDelta {
                sequence_number, ..
            }
            | Self::OutputTextDone {
                sequence_number, ..
            }
            | Self::ReasoningSummaryPartAdded {
                sequence_number, ..
            }
            | Self::ReasoningSummaryPartDone {
                sequence_number, ..
            }
            | Self::ReasoningSummaryTextDelta {
                sequence_number, ..
            }
            | Self::ReasoningSummaryTextDone {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDelta {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDone {
                sequence_number, ..
            }
            | Self::ResponseCompleted {
                sequence_number, ..
            }
            | Self::ResponseFailed {
                sequence_number, ..
            }
            | Self::ResponseIncomplete {
                sequence_number, ..
            } => *sequence_number,
            Self::Unknown => 0,
        }
    }

    /// The event's `type` string.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ResponseCreated { .. } => "response.created",
            Self::ResponseInProgress { .. } => "response.in_progress",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::OutputTextDone { .. } => "response.output_text.done",
            Self::ReasoningSummaryPartAdded { .. } => "response.reasoning_summary_part.added",
            Self::ReasoningSummaryPartDone { .. } => "response.reasoning_summary_part.done",
            Self::ReasoningSummaryTextDelta { .. } => "response.reasoning_summary_text.delta",
            Self::ReasoningSummaryTextDone { .. } => "response.reasoning_summary_text.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
            Self::ResponseCompleted { .. } => "response.completed",
            Self::ResponseFailed { .. } => "response.failed",
            Self::ResponseIncomplete { .. } => "response.incomplete",
            Self::Unknown => "unknown",
        }
    }
}

/// Translate one wire event into zero or more model stream events.
pub fn translate(event: StreamEvent) -> Vec<Result<ModelResponseStreamEvent, ModelError>> {
    match event {
        StreamEvent::Unknown => Vec::new(),
        StreamEvent::ResponseCreated { .. } | StreamEvent::ResponseInProgress { .. } => Vec::new(),

        StreamEvent::OutputItemAdded {
            output_index, item, ..
        } => {
            vec![Ok(ModelResponseStreamEvent::PartStart(
                PartStartEvent::new(output_index as usize, part_from_item(&item)),
            ))]
        }

        StreamEvent::OutputItemDone { output_index, .. } => {
            vec![Ok(ModelResponseStreamEvent::PartEnd(PartEndEvent::new(
                output_index as usize,
            )))]
        }

        StreamEvent::ContentPartAdded { .. } | StreamEvent::ContentPartDone { .. } => Vec::new(),

        StreamEvent::OutputTextDelta {
            output_index,
            delta,
            ..
        } => vec![Ok(ModelResponseStreamEvent::PartDelta(
            PartDeltaEvent::new(
                output_index as usize,
                ModelResponsePartDelta::Text(TextPartDelta {
                    content_delta: delta,
                    provider_details: None,
                }),
            ),
        ))],

        StreamEvent::OutputTextDone { .. } => Vec::new(),

        StreamEvent::ReasoningSummaryPartAdded { .. }
        | StreamEvent::ReasoningSummaryPartDone { .. } => Vec::new(),

        StreamEvent::ReasoningSummaryTextDelta {
            output_index,
            delta,
            ..
        } => vec![Ok(ModelResponseStreamEvent::PartDelta(
            PartDeltaEvent::new(
                output_index as usize,
                ModelResponsePartDelta::Thinking(ThinkingPartDelta {
                    content_delta: delta,
                    signature_delta: None,
                    provider_name: None,
                    provider_details: None,
                }),
            ),
        ))],

        StreamEvent::ReasoningSummaryTextDone { .. } => Vec::new(),

        StreamEvent::FunctionCallArgumentsDelta {
            output_index,
            delta,
            ..
        } => vec![Ok(ModelResponseStreamEvent::PartDelta(
            PartDeltaEvent::new(
                output_index as usize,
                ModelResponsePartDelta::ToolCall(ToolCallPartDelta {
                    args_delta: delta,
                    tool_call_id: None,
                    provider_details: None,
                }),
            ),
        ))],

        StreamEvent::FunctionCallArgumentsDone { .. } => Vec::new(),

        StreamEvent::ResponseCompleted { response, .. } => {
            vec![Ok(ModelResponseStreamEvent::StreamComplete(
                stream_complete(&response, FinishReason::Stop),
            ))]
        }

        StreamEvent::ResponseIncomplete { response, .. } => {
            vec![Ok(ModelResponseStreamEvent::StreamComplete(
                stream_complete(&response, FinishReason::Length),
            ))]
        }

        StreamEvent::ResponseFailed { response, .. } => vec![Err(failure(&response))],
    }
}

/// The initial part for an output item.
pub fn part_from_item(item: &OutputItem) -> ModelResponsePart {
    match item {
        OutputItem::Message { .. } => ModelResponsePart::Text(TextPart::new("")),
        OutputItem::Reasoning { .. } => ModelResponsePart::Thinking(ThinkingPart::new("")),
        OutputItem::FunctionCall { name, call_id, .. } => {
            ModelResponsePart::ToolCall(ToolCallPart {
                tool_name: name.clone(),
                args: ToolCallArgs::String(String::new()),
                tool_call_id: Some(call_id.clone()),
                id: None,
                provider_details: None,
            })
        }
    }
}

/// Build the terminal event with usage mapped from the response object.
pub fn stream_complete(response: &ResponseObject, reason: FinishReason) -> StreamCompleteEvent {
    let (input_tokens, output_tokens) = match &response.usage {
        Some(usage) => (usage.input_tokens, usage.output_tokens),
        None => (None, None),
    };
    StreamCompleteEvent {
        finish_reason: reason,
        input_tokens,
        output_tokens,
        cache_creation_tokens: None,
        cache_read_tokens: None,
    }
}

/// Map a `response.failed` object onto a model error.
pub fn failure(response: &ResponseObject) -> ModelError {
    let body = response.error.as_ref();
    let code = body
        .map(|error| error.code.clone())
        .unwrap_or_else(|| "response_failed".to_string());
    ModelError::provider(
        "openai",
        code.clone(),
        body.map(|error| error.message.clone())
            .unwrap_or_else(|| "response failed".to_string()),
        failure_kind(&code),
        None,
    )
}

/// Classify a wire error code for retry/fallback policies.
pub(super) fn failure_kind(code: &str) -> ModelFailureKind {
    match code {
        codes::WEBSOCKET_CONNECTION_LIMIT_REACHED => ModelFailureKind::RateLimited,
        codes::NOT_FOUND_ERROR | codes::PREVIOUS_RESPONSE_NOT_FOUND => ModelFailureKind::NotFound,
        codes::INVALID_REQUEST_ERROR => ModelFailureKind::InvalidRequest,
        _ => ModelFailureKind::Server,
    }
}

#[cfg(test)]
mod tests {
    use super::super::wire::{OutputContent, OutputItemStatus, ResponseUsage};
    use super::*;

    #[test]
    fn item_add_starts_parts() {
        let message = OutputItem::Message {
            id: "msg_1".into(),
            role: "assistant".to_string(),
            status: OutputItemStatus::InProgress,
            content: vec![OutputContent::OutputText {
                text: String::new(),
                annotations: Vec::new(),
            }],
        };
        let events = translate(StreamEvent::OutputItemAdded {
            sequence_number: 1,
            output_index: 0,
            item: message.clone(),
        });
        assert!(matches!(
            events[0],
            Ok(ModelResponseStreamEvent::PartStart(_))
        ));

        let call = OutputItem::FunctionCall {
            id: "fc_1".into(),
            call_id: "call_1".into(),
            name: "get_weather".into(),
            arguments: String::new(),
            status: OutputItemStatus::InProgress,
        };
        let events = translate(StreamEvent::OutputItemAdded {
            sequence_number: 2,
            output_index: 1,
            item: call,
        });
        match &events[0] {
            Ok(ModelResponseStreamEvent::PartStart(start)) => {
                assert!(matches!(
                    start.part,
                    ModelResponsePart::ToolCall(ref call)
                        if call.tool_name == "get_weather" && call.tool_call_id.as_deref() == Some("call_1")
                ));
            }
            other => panic!("expected part start, got {other:?}"),
        }
    }

    #[test]
    fn completed_maps_usage_and_finish_reason() {
        let response = ResponseObject {
            id: "resp_1".into(),
            usage: Some(ResponseUsage {
                input_tokens: Some(7),
                output_tokens: Some(4),
                total_tokens: Some(11),
            }),
            ..response_fixture()
        };
        let events = translate(StreamEvent::ResponseCompleted {
            sequence_number: 9,
            response,
        });
        match &events[0] {
            Ok(ModelResponseStreamEvent::StreamComplete(complete)) => {
                assert_eq!(complete.finish_reason, FinishReason::Stop);
                assert_eq!(complete.input_tokens, Some(7));
                assert_eq!(complete.output_tokens, Some(4));
            }
            other => panic!("expected stream complete, got {other:?}"),
        }
    }

    fn response_fixture() -> ResponseObject {
        use super::super::wire::CreateResponseRequest;
        ResponseObject::in_progress(
            "resp_1",
            0,
            "gpt-4o",
            &CreateResponseRequest {
                model: "gpt-4o".into(),
                input: super::super::wire::ResponseInput::Text(String::new()),
                instructions: None,
                tools: None,
                tool_choice: None,
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                stream: None,
                background: None,
                store: None,
                previous_response_id: None,
                reasoning: None,
                parallel_tool_calls: None,
                metadata: None,
                user: None,
                truncation: None,
                include: None,
                text: None,
                service_tier: None,
            },
        )
    }
}
