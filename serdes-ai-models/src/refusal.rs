//! Provider-neutral refusal contract.
//!
//! A provider that reports *why* it stopped - Anthropic sends
//! `stop_reason: "refusal"` together with `stop_details` - maps the stop to
//! [`FinishReason::ContentFilter`] and places the details, under the keys of
//! this module, into
//!
//! - the `details` of the terminal [`TerminalMetadata`] of a stream, which the
//!   agent and [`collect_stream`](crate::collect_stream) copy into
//!   [`ModelResponse::vendor_details`], and
//! - [`ModelResponse::vendor_details`] of a non-streamed response.
//!
//! The details object has the shape
//! `{"stop_reason": "...", "stop_details": {"type": "...", "category": "..."},
//! "input_transformations": ...}`; it is present only when the provider sent
//! `stop_details` or `input_transformations`.
//!
//! Callers read it back with [`StopDetails::from_response`] and
//! [`input_transformations`]. A refusal that leaves no answer becomes
//! [`ModelError::ContentFiltered`] in the agent, with a message built by
//! [`refusal_message`]; [`StopDetails::from_error_message`] reads it back.
//!
//! `stop_details.explanation` is never read in: [`StopDetails`] has no field for
//! it, so it can reach neither a log nor an error message.
//!
//! [`TerminalMetadata`]: serdes_ai_core::messages::TerminalMetadata

use crate::error::ModelError;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as JsonValue;
use serdes_ai_core::{FinishReason, ModelResponse, ModelResponsePart};

/// Key of the provider's raw stop reason in the details object.
pub const STOP_REASON_KEY: &str = "stop_reason";
/// Key of the [`StopDetails`] in the details object.
pub const STOP_DETAILS_KEY: &str = "stop_details";
/// Key of the raw `input_transformations` JSON in the details object.
pub const INPUT_TRANSFORMATIONS_KEY: &str = "input_transformations";
/// Prefix of every message built by [`refusal_message`].
pub const REFUSAL_MESSAGE_PREFIX: &str = "refusal:";

/// Placeholder for a value the provider did not send.
const UNKNOWN: &str = "unknown";

/// Why the provider stopped, as far as it is safe to keep.
///
/// Only `type` and `category` are read; `explanation` is dropped on purpose.
/// Both fields are lenient: a missing, `null` or non-string value becomes
/// `None` instead of failing the whole event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopDetails {
    /// The `type` of the stop details (for example `"refusal"`).
    #[serde(
        rename = "type",
        default,
        deserialize_with = "lenient_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub stop_type: Option<String>,
    /// The refusal category (for example `"reasoning_extraction"`).
    #[serde(
        default,
        deserialize_with = "lenient_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub category: Option<String>,
}

impl StopDetails {
    /// Read stop details from a raw JSON value. `None` unless it is an object.
    #[must_use]
    pub fn from_json(value: &JsonValue) -> Option<Self> {
        if !value.is_object() {
            return None;
        }
        serde_json::from_value(value.clone()).ok()
    }

    /// Read stop details from a details object (terminal metadata `details`
    /// or `vendor_details`) built by [`terminal_details`].
    #[must_use]
    pub fn from_details(details: &JsonValue) -> Option<Self> {
        details.get(STOP_DETAILS_KEY).and_then(Self::from_json)
    }

    /// Read stop details from the `vendor_details` of a response.
    #[must_use]
    pub fn from_response(response: &ModelResponse) -> Option<Self> {
        response
            .vendor_details
            .as_ref()
            .and_then(Self::from_details)
    }

    /// Read stop details back from a message built by [`refusal_message`] -
    /// also when it is embedded in a longer text, such as the `Display` of
    /// [`ModelError::ContentFiltered`] or `AgentStreamEvent::Error`.
    ///
    /// `None` when the text holds no refusal message.
    #[must_use]
    pub fn from_error_message(message: &str) -> Option<Self> {
        let start = message.find(REFUSAL_MESSAGE_PREFIX)?;
        let rest = &message[start + REFUSAL_MESSAGE_PREFIX.len()..];
        let mut details = Self::default();
        for token in rest.split_whitespace() {
            let Some((key, value)) = token.split_once('=') else {
                continue;
            };
            let value = (value != UNKNOWN).then(|| value.to_string());
            match key {
                "type" => details.stop_type = value,
                "category" => details.category = value,
                _ => {}
            }
        }
        Some(details)
    }
}

/// Accept any JSON value and keep it only when it is a string.
fn lenient_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<JsonValue>::deserialize(deserializer)?;
    Ok(value.and_then(|v| v.as_str().map(str::to_owned)))
}

/// Deserialize a wire `stop_details` field leniently: any shape the provider
/// sends is accepted, and only an object yields [`StopDetails`]. For use with
/// `#[serde(default, deserialize_with = "...")]` in provider wire types.
///
/// # Errors
///
/// Only when the input is not valid JSON at all.
pub fn lenient_stop_details<'de, D>(deserializer: D) -> Result<Option<StopDetails>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<JsonValue>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(StopDetails::from_json))
}

/// Build the details object for terminal metadata or `vendor_details`.
///
/// `None` when the provider sent neither stop details nor input
/// transformations, so a normal response keeps its metadata unchanged.
#[must_use]
pub fn terminal_details(
    stop_reason: Option<&str>,
    stop_details: Option<&StopDetails>,
    input_transformations: Option<&JsonValue>,
) -> Option<JsonValue> {
    if stop_details.is_none() && input_transformations.is_none() {
        return None;
    }
    let mut details = serde_json::Map::new();
    if let Some(reason) = stop_reason {
        details.insert(STOP_REASON_KEY.into(), JsonValue::from(reason));
    }
    if let Some(stop) = stop_details {
        details.insert(
            STOP_DETAILS_KEY.into(),
            serde_json::to_value(stop).unwrap_or(JsonValue::Null),
        );
    }
    if let Some(transformations) = input_transformations {
        details.insert(INPUT_TRANSFORMATIONS_KEY.into(), transformations.clone());
    }
    Some(JsonValue::Object(details))
}

/// The raw `input_transformations` JSON the provider returned, if any.
#[must_use]
pub fn input_transformations(response: &ModelResponse) -> Option<&JsonValue> {
    response
        .vendor_details
        .as_ref()
        .and_then(|details| details.get(INPUT_TRANSFORMATIONS_KEY))
}

/// The stable refusal message:
/// `refusal: type=<type> category=<category> finish_reason=<finish_reason>`.
///
/// A value the provider did not send reads `unknown`; every character outside
/// `[A-Za-z0-9_.:-]` is replaced by `_`, so the message stays one parseable
/// line and cannot forge a log entry.
#[must_use]
pub fn refusal_message(finish_reason: FinishReason, details: &StopDetails) -> String {
    format!(
        "{REFUSAL_MESSAGE_PREFIX} type={} category={} finish_reason={finish_reason}",
        sanitize(details.stop_type.as_deref()),
        sanitize(details.category.as_deref()),
    )
}

/// [`ModelError::ContentFiltered`] carrying the [`refusal_message`].
#[must_use]
pub fn refusal_error(finish_reason: FinishReason, details: &StopDetails) -> ModelError {
    ModelError::ContentFiltered(refusal_message(finish_reason, details))
}

/// The stop details of a refusal that left no answer.
///
/// `Some` when the finish reason is [`FinishReason::ContentFilter`], the
/// details object holds stop details, and the parts hold no answer - no
/// non-blank text, no tool call, no builtin tool call and no file; thinking
/// parts do not count as an answer. A refusal after real text yields `None`:
/// that response completes, carrying the stop details in `vendor_details`.
#[must_use]
pub fn refusal_without_answer(
    finish_reason: FinishReason,
    details: Option<&JsonValue>,
    parts: &[ModelResponsePart],
) -> Option<StopDetails> {
    if finish_reason != FinishReason::ContentFilter {
        return None;
    }
    let stop_details = details.and_then(StopDetails::from_details)?;
    let has_answer = parts.iter().any(|part| match part {
        ModelResponsePart::Text(text) => !text.content.trim().is_empty(),
        ModelResponsePart::Thinking(_) => false,
        ModelResponsePart::ToolCall(_)
        | ModelResponsePart::BuiltinToolCall(_)
        | ModelResponsePart::File(_) => true,
    });
    (!has_answer).then_some(stop_details)
}

fn sanitize(value: Option<&str>) -> String {
    match value {
        Some(v) if !v.is_empty() => v
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .collect(),
        _ => UNKNOWN.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_core::{TextPart, ThinkingPart, ToolCallPart};

    fn refusal_details() -> StopDetails {
        StopDetails {
            stop_type: Some("refusal".into()),
            category: Some("reasoning_extraction".into()),
        }
    }

    #[test]
    fn stop_details_drop_explanation_and_tolerate_odd_shapes() {
        let full = serde_json::json!({
            "type": "refusal", "category": "reasoning_extraction", "explanation": "secret"
        });
        let parsed = StopDetails::from_json(&full).unwrap();
        assert_eq!(parsed, refusal_details());
        assert!(!serde_json::to_string(&parsed).unwrap().contains("secret"));

        let odd = serde_json::json!({"category": null, "type": 7, "explanation": "secret"});
        assert_eq!(StopDetails::from_json(&odd), Some(StopDetails::default()));
        assert_eq!(StopDetails::from_json(&serde_json::json!("refusal")), None);
    }

    #[test]
    fn refusal_message_has_the_documented_format_and_round_trips() {
        let message = refusal_message(FinishReason::ContentFilter, &refusal_details());
        assert_eq!(
            message,
            "refusal: type=refusal category=reasoning_extraction finish_reason=content_filter"
        );
        let error = refusal_error(FinishReason::ContentFilter, &refusal_details());
        assert_eq!(
            StopDetails::from_error_message(&error.to_string()),
            Some(refusal_details())
        );
        assert_eq!(
            StopDetails::from_error_message("Content filtered: other"),
            None
        );
    }

    #[test]
    fn refusal_message_marks_missing_values_and_sanitizes() {
        let details = StopDetails {
            stop_type: None,
            category: Some("bad value\nforged=line".into()),
        };
        let message = refusal_message(FinishReason::ContentFilter, &details);
        assert_eq!(
            message,
            "refusal: type=unknown category=bad_value_forged_line finish_reason=content_filter"
        );
        let parsed = StopDetails::from_error_message(&message).unwrap();
        assert_eq!(parsed.stop_type, None);
        assert_eq!(parsed.category.as_deref(), Some("bad_value_forged_line"));
    }

    #[test]
    fn terminal_details_is_none_without_stop_details_or_transformations() {
        assert_eq!(terminal_details(Some("end_turn"), None, None), None);

        let details = terminal_details(Some("refusal"), Some(&refusal_details()), None).unwrap();
        assert_eq!(
            details,
            serde_json::json!({
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "reasoning_extraction"}
            })
        );

        let transformations = serde_json::json!([{"type": "drop_block"}]);
        let mut response = ModelResponse::new();
        response.vendor_details = terminal_details(None, None, Some(&transformations));
        assert_eq!(input_transformations(&response), Some(&transformations));
        assert_eq!(StopDetails::from_response(&response), None);
    }

    #[test]
    fn refusal_without_answer_ignores_thinking_but_not_text_or_tools() {
        let details = terminal_details(Some("refusal"), Some(&refusal_details()), None);
        let thinking = vec![ModelResponsePart::Thinking(
            ThinkingPart::new("").with_signature("sig"),
        )];
        assert_eq!(
            refusal_without_answer(FinishReason::ContentFilter, details.as_ref(), &thinking),
            Some(refusal_details())
        );
        assert_eq!(
            refusal_without_answer(FinishReason::ContentFilter, details.as_ref(), &[]),
            Some(refusal_details())
        );

        let blank = vec![ModelResponsePart::Text(TextPart::new("  "))];
        assert!(
            refusal_without_answer(FinishReason::ContentFilter, details.as_ref(), &blank).is_some()
        );

        let text = vec![ModelResponsePart::Text(TextPart::new("partial"))];
        assert_eq!(
            refusal_without_answer(FinishReason::ContentFilter, details.as_ref(), &text),
            None
        );
        let tool = vec![ModelResponsePart::ToolCall(ToolCallPart::new(
            "search",
            serde_json::json!({}),
        ))];
        assert_eq!(
            refusal_without_answer(FinishReason::ContentFilter, details.as_ref(), &tool),
            None
        );

        assert_eq!(
            refusal_without_answer(FinishReason::EndTurn, details.as_ref(), &[]),
            None
        );
        assert_eq!(
            refusal_without_answer(FinishReason::ContentFilter, None, &[]),
            None
        );
    }
}
