//! Streaming events for model responses.
//!
//! This module defines the event types used during streaming responses,
//! including part start, delta, and end events.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::parts::{
    BuiltinToolCallPart, FilePart, TextPart, ThinkingPart, ToolCallArgs, ToolCallPart,
};
use super::response::ModelResponsePart;

/// Stream event for model responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event_kind", rename_all = "snake_case")]
pub enum ModelResponseStreamEvent {
    /// A new part has started.
    PartStart(PartStartEvent),
    /// Delta for an existing part.
    PartDelta(PartDeltaEvent),
    /// A part has ended.
    PartEnd(PartEndEvent),
}

impl ModelResponseStreamEvent {
    /// Create a part start event.
    #[must_use]
    pub fn part_start(index: usize, part: ModelResponsePart) -> Self {
        Self::PartStart(PartStartEvent { index, part })
    }

    /// Create a text delta event.
    #[must_use]
    pub fn text_delta(index: usize, content_delta: impl Into<String>) -> Self {
        Self::PartDelta(PartDeltaEvent {
            index,
            delta: ModelResponsePartDelta::Text(TextPartDelta::new(content_delta)),
        })
    }

    /// Create a tool call delta event.
    #[must_use]
    pub fn tool_call_delta(index: usize, args_delta: impl Into<String>) -> Self {
        Self::PartDelta(PartDeltaEvent {
            index,
            delta: ModelResponsePartDelta::ToolCall(ToolCallPartDelta::new(args_delta)),
        })
    }

    /// Create a thinking delta event.
    #[must_use]
    pub fn thinking_delta(index: usize, content_delta: impl Into<String>) -> Self {
        Self::PartDelta(PartDeltaEvent {
            index,
            delta: ModelResponsePartDelta::Thinking(ThinkingPartDelta::new(content_delta)),
        })
    }

    /// Create a builtin tool call delta event.
    #[must_use]
    pub fn builtin_tool_call_delta(index: usize, args_delta: impl Into<String>) -> Self {
        Self::PartDelta(PartDeltaEvent {
            index,
            delta: ModelResponsePartDelta::BuiltinToolCall(BuiltinToolCallPartDelta::new(
                args_delta,
            )),
        })
    }

    /// Create a file part start event.
    ///
    /// Files arrive complete (no deltas), so this creates a start event
    /// with the full file content.
    #[must_use]
    pub fn file_part(index: usize, part: FilePart) -> Self {
        Self::PartStart(PartStartEvent {
            index,
            part: ModelResponsePart::File(part),
        })
    }

    /// Create a builtin tool call start event.
    #[must_use]
    pub fn builtin_tool_call_start(index: usize, part: BuiltinToolCallPart) -> Self {
        Self::PartStart(PartStartEvent {
            index,
            part: ModelResponsePart::BuiltinToolCall(part),
        })
    }

    /// Create a part end event.
    #[must_use]
    pub fn part_end(index: usize) -> Self {
        Self::PartEnd(PartEndEvent { index })
    }

    /// Get the part index.
    #[must_use]
    pub fn index(&self) -> usize {
        match self {
            Self::PartStart(e) => e.index,
            Self::PartDelta(e) => e.index,
            Self::PartEnd(e) => e.index,
        }
    }

    /// Check if this is a start event.
    #[must_use]
    pub fn is_start(&self) -> bool {
        matches!(self, Self::PartStart(_))
    }

    /// Check if this is a delta event.
    #[must_use]
    pub fn is_delta(&self) -> bool {
        matches!(self, Self::PartDelta(_))
    }

    /// Check if this is an end event.
    #[must_use]
    pub fn is_end(&self) -> bool {
        matches!(self, Self::PartEnd(_))
    }
}

/// Event indicating a new part has started.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartStartEvent {
    /// Index of the part in the response.
    pub index: usize,
    /// The initial part data.
    pub part: ModelResponsePart,
}

impl PartStartEvent {
    /// Create a new part start event.
    #[must_use]
    pub fn new(index: usize, part: ModelResponsePart) -> Self {
        Self { index, part }
    }

    /// Create a text part start.
    #[must_use]
    pub fn text(index: usize, content: impl Into<String>) -> Self {
        Self::new(index, ModelResponsePart::Text(TextPart::new(content)))
    }

    /// Create a tool call part start.
    #[must_use]
    pub fn tool_call(index: usize, tool_name: impl Into<String>) -> Self {
        Self::new(
            index,
            ModelResponsePart::ToolCall(ToolCallPart::new(tool_name, serde_json::Value::Null)),
        )
    }

    /// Create a thinking part start.
    #[must_use]
    pub fn thinking(index: usize, content: impl Into<String>) -> Self {
        Self::new(
            index,
            ModelResponsePart::Thinking(ThinkingPart::new(content)),
        )
    }

    /// Create a file part start.
    #[must_use]
    pub fn file(index: usize, part: FilePart) -> Self {
        Self::new(index, ModelResponsePart::File(part))
    }

    /// Create a builtin tool call part start.
    #[must_use]
    pub fn builtin_tool_call(
        index: usize,
        tool_name: impl Into<String>,
        args: impl Into<ToolCallArgs>,
    ) -> Self {
        Self::new(
            index,
            ModelResponsePart::BuiltinToolCall(BuiltinToolCallPart::new(tool_name, args)),
        )
    }
}

/// Event containing a delta update for a part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartDeltaEvent {
    /// Index of the part being updated.
    pub index: usize,
    /// The delta content.
    pub delta: ModelResponsePartDelta,
}

impl PartDeltaEvent {
    /// Create a new delta event.
    #[must_use]
    pub fn new(index: usize, delta: ModelResponsePartDelta) -> Self {
        Self { index, delta }
    }

    /// Create a text delta.
    #[must_use]
    pub fn text(index: usize, content: impl Into<String>) -> Self {
        Self::new(
            index,
            ModelResponsePartDelta::Text(TextPartDelta::new(content)),
        )
    }

    /// Create a tool call args delta.
    #[must_use]
    pub fn tool_call_args(index: usize, args: impl Into<String>) -> Self {
        Self::new(
            index,
            ModelResponsePartDelta::ToolCall(ToolCallPartDelta::new(args)),
        )
    }

    /// Create a thinking delta.
    #[must_use]
    pub fn thinking(index: usize, content: impl Into<String>) -> Self {
        Self::new(
            index,
            ModelResponsePartDelta::Thinking(ThinkingPartDelta::new(content)),
        )
    }

    /// Create a builtin tool call args delta.
    #[must_use]
    pub fn builtin_tool_call_args(index: usize, args: impl Into<String>) -> Self {
        Self::new(
            index,
            ModelResponsePartDelta::BuiltinToolCall(BuiltinToolCallPartDelta::new(args)),
        )
    }
}

/// Delta content for different part types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "delta_kind", rename_all = "snake_case")]
pub enum ModelResponsePartDelta {
    /// Text content delta.
    Text(TextPartDelta),
    /// Tool call arguments delta.
    ToolCall(ToolCallPartDelta),
    /// Thinking content delta.
    Thinking(ThinkingPartDelta),
    /// Builtin tool call arguments delta.
    BuiltinToolCall(BuiltinToolCallPartDelta),
}

impl ModelResponsePartDelta {
    /// Check if this is a text delta.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Check if this is a tool call delta.
    #[must_use]
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall(_))
    }

    /// Check if this is a thinking delta.
    #[must_use]
    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking(_))
    }

    /// Check if this is a builtin tool call delta.
    #[must_use]
    pub fn is_builtin_tool_call(&self) -> bool {
        matches!(self, Self::BuiltinToolCall(_))
    }

    /// Get the content delta if applicable.
    #[must_use]
    pub fn content_delta(&self) -> Option<&str> {
        match self {
            Self::Text(d) => Some(&d.content_delta),
            Self::Thinking(d) => Some(&d.content_delta),
            Self::ToolCall(_) => None,
            Self::BuiltinToolCall(_) => None,
        }
    }

    /// Apply this delta to a matching ModelResponsePart.
    ///
    /// Returns `true` if the delta was successfully applied (types matched),
    /// `false` if the types didn't match.
    #[must_use]
    pub fn apply(&self, part: &mut ModelResponsePart) -> bool {
        match (self, part) {
            (Self::Text(delta), ModelResponsePart::Text(text_part)) => {
                delta.apply(text_part);
                true
            }
            (Self::ToolCall(delta), ModelResponsePart::ToolCall(tool_part)) => {
                delta.apply(tool_part);
                true
            }
            (Self::Thinking(delta), ModelResponsePart::Thinking(thinking_part)) => {
                delta.apply(thinking_part);
                true
            }
            (Self::BuiltinToolCall(delta), ModelResponsePart::BuiltinToolCall(builtin_part)) => {
                delta.apply(builtin_part);
                true
            }
            _ => false,
        }
    }
}

/// Delta for text content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextPartDelta {
    /// The text content delta.
    pub content_delta: String,
    /// Provider-specific details/metadata delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<Map<String, Value>>,
}

impl TextPartDelta {
    /// Create a new text delta.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content_delta: content.into(),
            provider_details: None,
        }
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: Map<String, Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Check if the delta is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content_delta.is_empty() && self.provider_details.is_none()
    }

    /// Apply this delta to an existing TextPart.
    pub fn apply(&self, part: &mut TextPart) {
        if !self.content_delta.is_empty() {
            part.content.push_str(&self.content_delta);
        }
        if let Some(ref details) = self.provider_details {
            match &mut part.provider_details {
                Some(existing) => {
                    // Merge new details into existing
                    for (key, value) in details {
                        existing.insert(key.clone(), value.clone());
                    }
                }
                None => part.provider_details = Some(details.clone()),
            }
        }
    }
}

impl Default for TextPartDelta {
    fn default() -> Self {
        Self::new("")
    }
}

/// Delta for tool call arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallPartDelta {
    /// The arguments JSON delta.
    pub args_delta: String,
    /// Provider-assigned tool call ID (may arrive in delta).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Provider-specific details/metadata delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<Map<String, Value>>,
}

impl ToolCallPartDelta {
    /// Create a new tool call delta.
    #[must_use]
    pub fn new(args: impl Into<String>) -> Self {
        Self {
            args_delta: args.into(),
            tool_call_id: None,
            provider_details: None,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: Map<String, Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Check if the delta is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.args_delta.is_empty()
            && self.tool_call_id.is_none()
            && self.provider_details.is_none()
    }

    /// Apply this delta to an existing ToolCallPart.
    pub fn apply(&self, part: &mut ToolCallPart) {
        if !self.args_delta.is_empty() {
            // Check if current args are empty (just "{}") - if so, replace instead of append
            let current = part
                .args
                .to_json_string()
                .unwrap_or_else(|_| part.args.to_json().to_string());
            
            let new_args = if current == "{}" || current.is_empty() {
                // Start fresh with the delta
                self.args_delta.clone()
            } else {
                // Append to existing args
                format!("{}{}", current, self.args_delta)
            };
            part.args = ToolCallArgs::String(new_args);
        }
        if self.tool_call_id.is_some() && part.tool_call_id.is_none() {
            part.tool_call_id = self.tool_call_id.clone();
        }
        if let Some(ref details) = self.provider_details {
            match &mut part.provider_details {
                Some(existing) => {
                    // Merge new details into existing
                    for (key, value) in details {
                        existing.insert(key.clone(), value.clone());
                    }
                }
                None => part.provider_details = Some(details.clone()),
            }
        }
    }
}

impl Default for ToolCallPartDelta {
    fn default() -> Self {
        Self::new("")
    }
}

/// Delta for thinking content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingPartDelta {
    /// The thinking content delta.
    pub content_delta: String,
    /// Signature delta (for Anthropic's thinking blocks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_delta: Option<String>,
    /// Provider name (set once, not accumulated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
    /// Provider-specific details/metadata delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<Map<String, Value>>,
}

impl ThinkingPartDelta {
    /// Create a new thinking delta.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content_delta: content.into(),
            signature_delta: None,
            provider_name: None,
            provider_details: None,
        }
    }

    /// Set the signature delta.
    #[must_use]
    pub fn with_signature_delta(mut self, sig: impl Into<String>) -> Self {
        self.signature_delta = Some(sig.into());
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: Map<String, Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Check if the delta is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content_delta.is_empty()
            && self.signature_delta.is_none()
            && self.provider_name.is_none()
            && self.provider_details.is_none()
    }

    /// Apply this delta to an existing ThinkingPart.
    pub fn apply(&self, part: &mut ThinkingPart) {
        if !self.content_delta.is_empty() {
            part.content.push_str(&self.content_delta);
        }
        if let Some(ref sig_delta) = self.signature_delta {
            match &mut part.signature {
                Some(existing) => existing.push_str(sig_delta),
                None => part.signature = Some(sig_delta.clone()),
            }
        }
        if self.provider_name.is_some() {
            part.provider_name = self.provider_name.clone();
        }
        if let Some(ref details) = self.provider_details {
            match &mut part.provider_details {
                Some(existing) => {
                    // Merge new details into existing
                    for (key, value) in details {
                        existing.insert(key.clone(), value.clone());
                    }
                }
                None => part.provider_details = Some(details.clone()),
            }
        }
    }
}

impl Default for ThinkingPartDelta {
    fn default() -> Self {
        Self::new("")
    }
}

/// Delta for builtin tool call arguments.
///
/// Similar to `ToolCallPartDelta` but for builtin tools like web search,
/// code execution, and file search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuiltinToolCallPartDelta {
    /// The arguments JSON delta.
    pub args_delta: String,
    /// Provider-assigned tool call ID (may arrive in delta).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Provider-specific details/metadata delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_details: Option<Map<String, Value>>,
}

impl BuiltinToolCallPartDelta {
    /// Create a new builtin tool call delta.
    #[must_use]
    pub fn new(args: impl Into<String>) -> Self {
        Self {
            args_delta: args.into(),
            tool_call_id: None,
            provider_details: None,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set provider-specific details.
    #[must_use]
    pub fn with_provider_details(mut self, details: Map<String, Value>) -> Self {
        self.provider_details = Some(details);
        self
    }

    /// Check if the delta is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.args_delta.is_empty()
            && self.tool_call_id.is_none()
            && self.provider_details.is_none()
    }

    /// Apply this delta to an existing BuiltinToolCallPart.
    pub fn apply(&self, part: &mut BuiltinToolCallPart) {
        if !self.args_delta.is_empty() {
            // Check if current args are empty (just "{}") - if so, replace instead of append
            let current = part
                .args
                .to_json_string()
                .unwrap_or_else(|_| part.args.to_json().to_string());
            
            let new_args = if current == "{}" || current.is_empty() {
                // Start fresh with the delta
                self.args_delta.clone()
            } else {
                // Append to existing args
                format!("{}{}", current, self.args_delta)
            };
            part.args = ToolCallArgs::String(new_args);
        }
        if self.tool_call_id.is_some() && part.tool_call_id.is_none() {
            part.tool_call_id = self.tool_call_id.clone();
        }
        if let Some(ref details) = self.provider_details {
            match &mut part.provider_details {
                Some(existing) => {
                    // Merge new details into existing
                    for (key, value) in details {
                        existing.insert(key.clone(), value.clone());
                    }
                }
                None => part.provider_details = Some(details.clone()),
            }
        }
    }
}

impl Default for BuiltinToolCallPartDelta {
    fn default() -> Self {
        Self::new("")
    }
}

/// Event indicating a part has ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartEndEvent {
    /// Index of the part that ended.
    pub index: usize,
}

impl PartEndEvent {
    /// Create a new part end event.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part_start_event() {
        let event = PartStartEvent::text(0, "Hello");
        assert_eq!(event.index, 0);
        assert!(matches!(event.part, ModelResponsePart::Text(_)));
    }

    #[test]
    fn test_part_delta_event() {
        let event = PartDeltaEvent::text(0, " world");
        assert_eq!(event.index, 0);
        assert!(event.delta.is_text());
        assert_eq!(event.delta.content_delta(), Some(" world"));
    }

    #[test]
    fn test_stream_event_helpers() {
        let start = ModelResponseStreamEvent::part_start(
            0,
            ModelResponsePart::text("Hello"),
        );
        assert!(start.is_start());
        assert_eq!(start.index(), 0);

        let delta = ModelResponseStreamEvent::text_delta(0, " world");
        assert!(delta.is_delta());

        let end = ModelResponseStreamEvent::part_end(0);
        assert!(end.is_end());
    }

    #[test]
    fn test_serde_roundtrip() {
        let event = ModelResponseStreamEvent::text_delta(0, "Hello");
        let json = serde_json::to_string(&event).unwrap();
        let parsed: ModelResponseStreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn test_text_delta_apply() {
        let mut part = TextPart::new("Hello");
        let delta = TextPartDelta::new(" world");
        delta.apply(&mut part);
        assert_eq!(part.content, "Hello world");
    }

    #[test]
    fn test_text_delta_apply_with_provider_details() {
        let mut part = TextPart::new("Hello");
        
        let mut details = Map::new();
        details.insert("model".to_string(), Value::String("gpt-4".to_string()));
        
        let delta = TextPartDelta::new(" world").with_provider_details(details);
        delta.apply(&mut part);
        
        assert_eq!(part.content, "Hello world");
        assert!(part.provider_details.is_some());
        assert_eq!(
            part.provider_details.as_ref().unwrap().get("model"),
            Some(&Value::String("gpt-4".to_string()))
        );
    }

    #[test]
    fn test_text_delta_merge_provider_details() {
        let mut initial_details = Map::new();
        initial_details.insert("key1".to_string(), Value::String("value1".to_string()));
        
        let mut part = TextPart::new("Hello").with_provider_details(initial_details);
        
        let mut new_details = Map::new();
        new_details.insert("key2".to_string(), Value::String("value2".to_string()));
        
        let delta = TextPartDelta::new("").with_provider_details(new_details);
        delta.apply(&mut part);
        
        let details = part.provider_details.as_ref().unwrap();
        assert_eq!(details.len(), 2);
        assert_eq!(details.get("key1"), Some(&Value::String("value1".to_string())));
        assert_eq!(details.get("key2"), Some(&Value::String("value2".to_string())));
    }

    #[test]
    fn test_thinking_delta_apply() {
        let mut part = ThinkingPart::new("Initial thought");
        let delta = ThinkingPartDelta::new(" continued...");
        delta.apply(&mut part);
        assert_eq!(part.content, "Initial thought continued...");
    }

    #[test]
    fn test_thinking_delta_apply_with_signature() {
        let mut part = ThinkingPart::new("Thinking");
        
        let delta = ThinkingPartDelta::new("")
            .with_signature_delta("sig123")
            .with_provider_name("anthropic");
        delta.apply(&mut part);
        
        assert_eq!(part.signature, Some("sig123".to_string()));
        assert_eq!(part.provider_name, Some("anthropic".to_string()));
    }

    #[test]
    fn test_thinking_delta_signature_accumulation() {
        let mut part = ThinkingPart::new("Thinking").with_signature("sig1");
        
        let delta = ThinkingPartDelta::new("").with_signature_delta("23");
        delta.apply(&mut part);
        
        assert_eq!(part.signature, Some("sig123".to_string()));
    }

    #[test]
    fn test_tool_call_delta_apply() {
        let mut part = ToolCallPart::new("get_weather", serde_json::json!({}));
        
        let delta = ToolCallPartDelta::new(r#"{"city":"NYC"}"#)
            .with_tool_call_id("call_123");
        delta.apply(&mut part);
        
        assert_eq!(part.tool_call_id, Some("call_123".to_string()));
        // Args should have the delta appended
        assert!(part.args.to_json_string().unwrap().contains("city"));
    }

    #[test]
    fn test_tool_call_delta_doesnt_overwrite_id() {
        let mut part = ToolCallPart::new("search", serde_json::json!({}))
            .with_tool_call_id("original_id");
        
        let delta = ToolCallPartDelta::new("").with_tool_call_id("new_id");
        delta.apply(&mut part);
        
        // Should keep original ID
        assert_eq!(part.tool_call_id, Some("original_id".to_string()));
    }

    #[test]
    fn test_model_response_part_delta_apply() {
        let mut text_part = ModelResponsePart::Text(TextPart::new("Hello"));
        let delta = ModelResponsePartDelta::Text(TextPartDelta::new(" world"));
        
        assert!(delta.apply(&mut text_part));
        
        if let ModelResponsePart::Text(ref text) = text_part {
            assert_eq!(text.content, "Hello world");
        } else {
            panic!("Expected Text part");
        }
    }

    #[test]
    fn test_model_response_part_delta_apply_type_mismatch() {
        let mut text_part = ModelResponsePart::Text(TextPart::new("Hello"));
        let delta = ModelResponsePartDelta::Thinking(ThinkingPartDelta::new("thinking"));
        
        // Should return false for type mismatch
        assert!(!delta.apply(&mut text_part));
        
        // Part should be unchanged
        if let ModelResponsePart::Text(ref text) = text_part {
            assert_eq!(text.content, "Hello");
        } else {
            panic!("Expected Text part");
        }
    }

    #[test]
    fn test_delta_is_empty() {
        // Empty text delta
        let text_delta = TextPartDelta::default();
        assert!(text_delta.is_empty());
        
        // Non-empty text delta
        let text_delta = TextPartDelta::new("content");
        assert!(!text_delta.is_empty());
        
        // Text delta with only provider details
        let mut details = Map::new();
        details.insert("key".to_string(), Value::Null);
        let text_delta = TextPartDelta::new("").with_provider_details(details);
        assert!(!text_delta.is_empty());
        
        // Empty thinking delta
        let thinking_delta = ThinkingPartDelta::default();
        assert!(thinking_delta.is_empty());
        
        // Thinking delta with only signature
        let thinking_delta = ThinkingPartDelta::new("").with_signature_delta("sig");
        assert!(!thinking_delta.is_empty());
        
        // Empty tool call delta
        let tool_delta = ToolCallPartDelta::default();
        assert!(tool_delta.is_empty());
        
        // Tool call delta with only tool_call_id
        let tool_delta = ToolCallPartDelta::new("").with_tool_call_id("id");
        assert!(!tool_delta.is_empty());
    }

    #[test]
    fn test_delta_builders() {
        let text_delta = TextPartDelta::new("content")
            .with_provider_details(Map::new());
        assert!(!text_delta.is_empty());
        
        let thinking_delta = ThinkingPartDelta::new("thought")
            .with_signature_delta("sig")
            .with_provider_name("provider")
            .with_provider_details(Map::new());
        assert!(!thinking_delta.is_empty());
        
        let tool_delta = ToolCallPartDelta::new(r#"{}"#)
            .with_tool_call_id("call_1")
            .with_provider_details(Map::new());
        assert!(!tool_delta.is_empty());
    }

    #[test]
    fn test_serde_roundtrip_with_new_fields() {
        // Text delta with provider_details
        let mut details = Map::new();
        details.insert("key".to_string(), Value::String("value".to_string()));
        let delta = TextPartDelta::new("hello").with_provider_details(details);
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: TextPartDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(delta, parsed);
        
        // Thinking delta with all fields
        let thinking_delta = ThinkingPartDelta::new("thought")
            .with_signature_delta("sig")
            .with_provider_name("anthropic");
        let json = serde_json::to_string(&thinking_delta).unwrap();
        let parsed: ThinkingPartDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(thinking_delta, parsed);
        
        // Tool call delta with all fields
        let tool_delta = ToolCallPartDelta::new(r#"{"a":1}"#)
            .with_tool_call_id("call_123");
        let json = serde_json::to_string(&tool_delta).unwrap();
        let parsed: ToolCallPartDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(tool_delta, parsed);
    }

    #[test]
    fn test_serde_skip_none_fields() {
        // Verify that None fields are not serialized
        let delta = TextPartDelta::new("hello");
        let json = serde_json::to_string(&delta).unwrap();
        
        assert!(json.contains("content_delta"));
        assert!(!json.contains("provider_details"));
        
        let thinking_delta = ThinkingPartDelta::new("thought");
        let json = serde_json::to_string(&thinking_delta).unwrap();
        
        assert!(!json.contains("signature_delta"));
        assert!(!json.contains("provider_name"));
        assert!(!json.contains("provider_details"));
    }

    #[test]
    fn test_backward_compat_deserialization() {
        // Verify we can deserialize old JSON without the new fields
        let old_json = r#"{"content_delta":"hello"}"#;
        let delta: TextPartDelta = serde_json::from_str(old_json).unwrap();
        assert_eq!(delta.content_delta, "hello");
        assert!(delta.provider_details.is_none());
        
        let old_json = r#"{"content_delta":"thinking"}"#;
        let delta: ThinkingPartDelta = serde_json::from_str(old_json).unwrap();
        assert_eq!(delta.content_delta, "thinking");
        assert!(delta.signature_delta.is_none());
        assert!(delta.provider_name.is_none());
        
        let old_json = r#"{"args_delta":"{}"}"#;
        let delta: ToolCallPartDelta = serde_json::from_str(old_json).unwrap();
        assert_eq!(delta.args_delta, "{}");
        assert!(delta.tool_call_id.is_none());
    }

    #[test]
    fn test_builtin_tool_call_delta() {
        let delta = BuiltinToolCallPartDelta::new(r#"{"query":"rust"}"#)
            .with_tool_call_id("builtin_123");
        
        assert!(!delta.is_empty());
        assert_eq!(delta.args_delta, r#"{"query":"rust"}"#);
        assert_eq!(delta.tool_call_id, Some("builtin_123".to_string()));
    }

    #[test]
    fn test_builtin_tool_call_delta_apply() {
        let mut part = BuiltinToolCallPart::new("web_search", serde_json::json!({}));
        
        let delta = BuiltinToolCallPartDelta::new(r#"{"q":"test"}"#)
            .with_tool_call_id("call_456");
        delta.apply(&mut part);
        
        assert_eq!(part.tool_call_id, Some("call_456".to_string()));
        assert!(part.args.to_json_string().unwrap().contains("test"));
    }

    #[test]
    fn test_file_part_event() {
        let file = FilePart::from_bytes(vec![0x89, 0x50, 0x4E, 0x47], "image/png");
        let event = ModelResponseStreamEvent::file_part(0, file.clone());
        
        assert!(event.is_start());
        assert_eq!(event.index(), 0);
        
        if let ModelResponseStreamEvent::PartStart(start) = event {
            assert!(matches!(start.part, ModelResponsePart::File(_)));
        } else {
            panic!("Expected PartStart");
        }
    }

    #[test]
    fn test_builtin_tool_call_start_event() {
        let part = BuiltinToolCallPart::new("web_search", serde_json::json!({"query": "rust"}))
            .with_tool_call_id("call_123");
        let event = ModelResponseStreamEvent::builtin_tool_call_start(0, part);
        
        assert!(event.is_start());
        
        if let ModelResponseStreamEvent::PartStart(start) = event {
            assert!(matches!(start.part, ModelResponsePart::BuiltinToolCall(_)));
        } else {
            panic!("Expected PartStart");
        }
    }

    #[test]
    fn test_builtin_tool_call_delta_event() {
        let event = ModelResponseStreamEvent::builtin_tool_call_delta(0, r#"{"q":"rust"}"#);
        
        assert!(event.is_delta());
        
        if let ModelResponseStreamEvent::PartDelta(delta_event) = event {
            assert!(delta_event.delta.is_builtin_tool_call());
        } else {
            panic!("Expected PartDelta");
        }
    }

    #[test]
    fn test_part_start_event_builtin_tool_call() {
        let event = PartStartEvent::builtin_tool_call(0, "code_execution", serde_json::json!({"code": "print(1)"}));
        
        assert_eq!(event.index, 0);
        assert!(matches!(event.part, ModelResponsePart::BuiltinToolCall(_)));
    }

    #[test]
    fn test_part_delta_event_builtin_tool_call() {
        let event = PartDeltaEvent::builtin_tool_call_args(0, r#"{"more":"args"}"#);
        
        assert_eq!(event.index, 0);
        assert!(event.delta.is_builtin_tool_call());
        assert_eq!(event.delta.content_delta(), None); // Builtin tool calls have no content delta
    }

    #[test]
    fn test_serde_roundtrip_builtin_tool_call_delta() {
        let delta = BuiltinToolCallPartDelta::new(r#"{"q":"test"}"#)
            .with_tool_call_id("call_123");
        
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: BuiltinToolCallPartDelta = serde_json::from_str(&json).unwrap();
        
        assert_eq!(delta, parsed);
    }

    #[test]
    fn test_model_response_part_delta_apply_builtin() {
        let mut part = ModelResponsePart::BuiltinToolCall(
            BuiltinToolCallPart::new("search", serde_json::json!({}))
        );
        let delta = ModelResponsePartDelta::BuiltinToolCall(
            BuiltinToolCallPartDelta::new(r#"{"q":"rust"}"#)
        );
        
        assert!(delta.apply(&mut part));
        
        if let ModelResponsePart::BuiltinToolCall(ref builtin) = part {
            assert!(builtin.args.to_json_string().unwrap().contains("rust"));
        } else {
            panic!("Expected BuiltinToolCall part");
        }
    }
}
