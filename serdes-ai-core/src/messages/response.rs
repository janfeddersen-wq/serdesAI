//! Response message types from model interactions.
//!
//! This module defines the message types that are returned FROM the model,
//! including text content, tool calls, and thinking/reasoning content.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::parts::{BuiltinToolCallPart, FilePart, TextPart, ThinkingPart, ToolCallPart};
use crate::usage::RequestUsage;

/// A complete model response containing multiple parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The response parts.
    pub parts: Vec<ModelResponsePart>,
    /// Name of the model that generated this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// When this response was received.
    pub timestamp: DateTime<Utc>,
    /// Why the model stopped generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Token usage for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<RequestUsage>,
    /// Vendor-specific response ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_id: Option<String>,
    /// Vendor-specific details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_details: Option<serde_json::Value>,
    /// Kind identifier.
    #[serde(default = "default_response_kind")]
    pub kind: String,
}

fn default_response_kind() -> String {
    "response".to_string()
}

impl ModelResponse {
    /// Create a new empty response.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            model_name: None,
            timestamp: Utc::now(),
            finish_reason: None,
            usage: None,
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        }
    }

    /// Create a response with the given parts.
    #[must_use]
    pub fn with_parts(parts: Vec<ModelResponsePart>) -> Self {
        Self {
            parts,
            ..Self::new()
        }
    }

    /// Create a simple text response.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::with_parts(vec![ModelResponsePart::Text(TextPart::new(content))])
    }

    /// Add a part.
    pub fn add_part(&mut self, part: ModelResponsePart) {
        self.parts.push(part);
    }

    /// Set the model name.
    #[must_use]
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Set the finish reason.
    #[must_use]
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    /// Set the usage.
    #[must_use]
    pub fn with_usage(mut self, usage: RequestUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set the vendor ID.
    #[must_use]
    pub fn with_vendor_id(mut self, id: impl Into<String>) -> Self {
        self.vendor_id = Some(id.into());
        self
    }

    /// Set vendor details.
    #[must_use]
    pub fn with_vendor_details(mut self, details: serde_json::Value) -> Self {
        self.vendor_details = Some(details);
        self
    }

    /// Get all text parts.
    #[must_use]
    pub fn text_parts(&self) -> impl Iterator<Item = &TextPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelResponsePart::Text(t) => Some(t),
            _ => None,
        })
    }

    /// Get all tool call parts.
    #[must_use]
    pub fn tool_call_parts(&self) -> impl Iterator<Item = &ToolCallPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelResponsePart::ToolCall(t) => Some(t),
            _ => None,
        })
    }

    /// Get all thinking parts.
    #[must_use]
    pub fn thinking_parts(&self) -> impl Iterator<Item = &ThinkingPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelResponsePart::Thinking(t) => Some(t),
            _ => None,
        })
    }

    /// Get all file parts.
    #[must_use]
    pub fn file_parts(&self) -> impl Iterator<Item = &FilePart> {
        self.parts.iter().filter_map(|p| match p {
            ModelResponsePart::File(f) => Some(f),
            _ => None,
        })
    }

    /// Get all text parts as a vector.
    #[deprecated(note = "Use text_parts() iterator instead")]
    pub fn text_parts_vec(&self) -> Vec<&TextPart> {
        self.text_parts().collect()
    }

    /// Get all tool call parts as a vector.
    #[deprecated(note = "Use tool_call_parts() iterator instead")]
    pub fn tool_call_parts_vec(&self) -> Vec<&ToolCallPart> {
        self.tool_call_parts().collect()
    }

    /// Get all thinking parts as a vector.
    #[deprecated(note = "Use thinking_parts() iterator instead")]
    pub fn thinking_parts_vec(&self) -> Vec<&ThinkingPart> {
        self.thinking_parts().collect()
    }

    /// Get all file parts as a vector.
    #[deprecated(note = "Use file_parts() iterator instead")]
    pub fn file_parts_vec(&self) -> Vec<&FilePart> {
        self.file_parts().collect()
    }

    /// Check if this response contains file parts.
    #[must_use]
    pub fn has_files(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, ModelResponsePart::File(_)))
    }

    /// Get all builtin tool call parts.
    #[must_use]
    pub fn builtin_tool_call_parts(&self) -> impl Iterator<Item = &BuiltinToolCallPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelResponsePart::BuiltinToolCall(b) => Some(b),
            _ => None,
        })
    }

    /// Get all builtin tool call parts as a vector.
    #[deprecated(note = "Use builtin_tool_call_parts() iterator instead")]
    pub fn builtin_tool_call_parts_vec(&self) -> Vec<&BuiltinToolCallPart> {
        self.builtin_tool_call_parts().collect()
    }

    /// Check if this response contains builtin tool calls.
    #[must_use]
    pub fn has_builtin_tool_calls(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, ModelResponsePart::BuiltinToolCall(_)))
    }

    /// Get combined text content.
    #[must_use]
    pub fn text_content(&self) -> String {
        self.text_parts()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Check if this response contains tool calls.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.parts
            .iter()
            .any(|p| matches!(p, ModelResponsePart::ToolCall(_)))
    }

    /// Check if the response is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Get the number of parts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parts.len()
    }
}

impl Default for ModelResponse {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<ModelResponsePart> for ModelResponse {
    fn from_iter<T: IntoIterator<Item = ModelResponsePart>>(iter: T) -> Self {
        Self::with_parts(iter.into_iter().collect())
    }
}

/// Individual parts of a model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "part_kind", rename_all = "kebab-case")]
pub enum ModelResponsePart {
    /// Text content.
    Text(TextPart),
    /// Tool call.
    ToolCall(ToolCallPart),
    /// Thinking/reasoning content.
    Thinking(ThinkingPart),
    /// File content (e.g., generated images).
    File(FilePart),
    /// Builtin tool call (web search, code execution, etc.).
    BuiltinToolCall(BuiltinToolCallPart),
}

impl ModelResponsePart {
    /// Create a text part.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(TextPart::new(content))
    }

    /// Create a tool call part.
    #[must_use]
    pub fn tool_call(
        tool_name: impl Into<String>,
        args: impl Into<super::parts::ToolCallArgs>,
    ) -> Self {
        Self::ToolCall(ToolCallPart::new(tool_name, args))
    }

    /// Create a thinking part.
    #[must_use]
    pub fn thinking(content: impl Into<String>) -> Self {
        Self::Thinking(ThinkingPart::new(content))
    }

    /// Create a file part from raw bytes and media type.
    #[must_use]
    pub fn file(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::File(FilePart::from_bytes(data, media_type))
    }

    /// Create a builtin tool call part.
    #[must_use]
    pub fn builtin_tool_call(
        tool_name: impl Into<String>,
        args: impl Into<super::parts::ToolCallArgs>,
    ) -> Self {
        Self::BuiltinToolCall(BuiltinToolCallPart::new(tool_name, args))
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        match self {
            Self::Text(_) => TextPart::PART_KIND,
            Self::ToolCall(_) => ToolCallPart::PART_KIND,
            Self::Thinking(_) => ThinkingPart::PART_KIND,
            Self::File(_) => FilePart::PART_KIND,
            Self::BuiltinToolCall(_) => BuiltinToolCallPart::PART_KIND,
        }
    }

    /// Check if this is a text part.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Check if this is a tool call part.
    #[must_use]
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall(_))
    }

    /// Check if this is a thinking part.
    #[must_use]
    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking(_))
    }

    /// Check if this is a file part.
    #[must_use]
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File(_))
    }

    /// Check if this is a builtin tool call part.
    #[must_use]
    pub fn is_builtin_tool_call(&self) -> bool {
        matches!(self, Self::BuiltinToolCall(_))
    }
}

impl From<TextPart> for ModelResponsePart {
    fn from(p: TextPart) -> Self {
        Self::Text(p)
    }
}

impl From<ToolCallPart> for ModelResponsePart {
    fn from(p: ToolCallPart) -> Self {
        Self::ToolCall(p)
    }
}

impl From<ThinkingPart> for ModelResponsePart {
    fn from(p: ThinkingPart) -> Self {
        Self::Thinking(p)
    }
}

impl From<FilePart> for ModelResponsePart {
    fn from(p: FilePart) -> Self {
        Self::File(p)
    }
}

impl From<BuiltinToolCallPart> for ModelResponsePart {
    fn from(p: BuiltinToolCallPart) -> Self {
        Self::BuiltinToolCall(p)
    }
}

/// Reason why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of response.
    Stop,
    /// Maximum tokens reached.
    Length,
    /// Content was filtered.
    ContentFilter,
    /// Model wants to call tools.
    ToolCall,
    /// An error occurred.
    Error,
    /// End of turn.
    EndTurn,
    /// Stop sequence encountered.
    StopSequence,
}

impl FinishReason {
    /// Check if this indicates the response is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Stop | Self::EndTurn | Self::StopSequence)
    }

    /// Check if this indicates truncation.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Length)
    }

    /// Check if this indicates tool use.
    #[must_use]
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall)
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ContentFilter => write!(f, "content_filter"),
            Self::ToolCall => write!(f, "tool_call"),
            Self::Error => write!(f, "error"),
            Self::EndTurn => write!(f, "end_turn"),
            Self::StopSequence => write!(f, "stop_sequence"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_response_new() {
        let response = ModelResponse::new();
        assert!(response.is_empty());
        assert!(!response.has_tool_calls());
    }

    #[test]
    fn test_model_response_text() {
        let response = ModelResponse::text("Hello, world!");
        assert_eq!(response.len(), 1);
        assert_eq!(response.text_content(), "Hello, world!");
    }

    #[test]
    fn test_model_response_with_tool_calls() {
        let response = ModelResponse::with_parts(vec![
            ModelResponsePart::text("Let me check the weather."),
            ModelResponsePart::tool_call("get_weather", serde_json::json!({"city": "NYC"})),
        ]);
        assert!(response.has_tool_calls());
        assert_eq!(response.tool_call_parts().count(), 1);
    }

    #[test]
    fn test_finish_reason() {
        assert!(FinishReason::Stop.is_complete());
        assert!(FinishReason::Length.is_truncated());
        assert!(FinishReason::ToolCall.is_tool_call());
    }

    #[test]
    fn test_serde_roundtrip() {
        let response = ModelResponse::with_parts(vec![
            ModelResponsePart::text("Hello"),
            ModelResponsePart::thinking("Thinking..."),
        ])
        .with_model_name("gpt-4")
        .with_finish_reason(FinishReason::Stop);

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ModelResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.len(), parsed.len());
        assert_eq!(response.model_name, parsed.model_name);
    }
}
