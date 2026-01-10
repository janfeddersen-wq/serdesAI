//! Partial response accumulation.
//!
//! This module provides types for accumulating streaming deltas into
//! complete responses.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serdes_ai_core::messages::{TextPart, ThinkingPart, ToolCallArgs, ToolCallPart};
use serdes_ai_core::{FinishReason, ModelResponse, ModelResponsePart, RequestUsage};

/// Partial part being accumulated.
#[derive(Debug, Clone)]
enum PartialPart {
    /// Text content.
    Text { content: String },
    /// Tool call.
    ToolCall {
        name: Option<String>,
        args: String,
        id: Option<String>,
    },
    /// Thinking content.
    Thinking {
        content: String,
        signature: Option<String>,
    },
}

impl PartialPart {
    /// Create new text part.
    fn text() -> Self {
        Self::Text {
            content: String::new(),
        }
    }

    /// Create new tool call part.
    fn tool_call() -> Self {
        Self::ToolCall {
            name: None,
            args: String::new(),
            id: None,
        }
    }

    /// Create new thinking part.
    fn thinking() -> Self {
        Self::Thinking {
            content: String::new(),
            signature: None,
        }
    }

    /// Check if this part has any content.
    fn has_content(&self) -> bool {
        match self {
            Self::Text { content } => !content.is_empty(),
            Self::ToolCall { name, args, .. } => name.is_some() || !args.is_empty(),
            Self::Thinking { content, .. } => !content.is_empty(),
        }
    }
}

/// Accumulates streaming deltas into a complete response.
#[derive(Debug, Clone)]
pub struct PartialResponse {
    parts: Vec<PartialPart>,
    model_name: Option<String>,
    usage: Option<RequestUsage>,
    finish_reason: Option<FinishReason>,
    timestamp: DateTime<Utc>,
    vendor_id: Option<String>,
}

impl Default for PartialResponse {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialResponse {
    /// Create a new partial response.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            model_name: None,
            usage: None,
            finish_reason: None,
            timestamp: Utc::now(),
            vendor_id: None,
        }
    }

    /// Ensure we have at least `n` parts, expanding with default type.
    fn ensure_parts(&mut self, n: usize, default_fn: impl Fn() -> PartialPart) {
        while self.parts.len() <= n {
            self.parts.push(default_fn());
        }
    }

    /// Apply a text delta.
    pub fn apply_text_delta(&mut self, index: usize, content: &str) {
        self.ensure_parts(index, PartialPart::text);

        // Ensure it's a text part
        if !matches!(self.parts[index], PartialPart::Text { .. }) {
            self.parts[index] = PartialPart::text();
        }

        if let PartialPart::Text {
            content: existing, ..
        } = &mut self.parts[index]
        {
            existing.push_str(content);
        }
    }

    /// Apply a tool call delta.
    pub fn apply_tool_delta(
        &mut self,
        index: usize,
        name: Option<&str>,
        args_delta: Option<&str>,
        id: Option<&str>,
    ) {
        self.ensure_parts(index, PartialPart::tool_call);

        // Ensure it's a tool call part
        if !matches!(self.parts[index], PartialPart::ToolCall { .. }) {
            self.parts[index] = PartialPart::tool_call();
        }

        if let PartialPart::ToolCall {
            name: existing_name,
            args,
            id: existing_id,
        } = &mut self.parts[index]
        {
            if let Some(n) = name {
                *existing_name = Some(n.to_string());
            }
            if let Some(a) = args_delta {
                args.push_str(a);
            }
            if let Some(i) = id {
                *existing_id = Some(i.to_string());
            }
        }
    }

    /// Apply a thinking delta.
    pub fn apply_thinking_delta(&mut self, index: usize, content: &str, signature: Option<&str>) {
        self.ensure_parts(index, PartialPart::thinking);

        // Ensure it's a thinking part
        if !matches!(self.parts[index], PartialPart::Thinking { .. }) {
            self.parts[index] = PartialPart::thinking();
        }

        if let PartialPart::Thinking {
            content: existing,
            signature: existing_sig,
        } = &mut self.parts[index]
        {
            existing.push_str(content);
            if let Some(s) = signature {
                *existing_sig = Some(s.to_string());
            }
        }
    }

    /// Set the model name.
    pub fn set_model_name(&mut self, name: impl Into<String>) {
        self.model_name = Some(name.into());
    }

    /// Set the usage.
    pub fn set_usage(&mut self, usage: RequestUsage) {
        self.usage = Some(usage);
    }

    /// Set the finish reason.
    pub fn set_finish_reason(&mut self, reason: FinishReason) {
        self.finish_reason = Some(reason);
    }

    /// Set the vendor ID.
    pub fn set_vendor_id(&mut self, id: impl Into<String>) {
        self.vendor_id = Some(id.into());
    }

    /// Get accumulated text content.
    #[must_use]
    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                PartialPart::Text { content } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get the number of parts.
    #[must_use]
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Check if the response is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| !p.has_content())
    }

    /// Finalize into a complete response (consumes self).
    #[must_use]
    pub fn finalize(self) -> ModelResponse {
        let parts = self
            .parts
            .into_iter()
            .filter(|p| p.has_content())
            .filter_map(|p| match p {
                PartialPart::Text { content } => {
                    Some(ModelResponsePart::Text(TextPart::new(content)))
                }
                PartialPart::ToolCall {
                    name: Some(name),
                    args,
                    id,
                } => {
                    let parsed_args: JsonValue =
                        serde_json::from_str(&args).unwrap_or(JsonValue::Null);
                    let mut tc = ToolCallPart::new(name, ToolCallArgs::Json(parsed_args));
                    if let Some(id) = id {
                        tc.tool_call_id = Some(id);
                    }
                    Some(ModelResponsePart::ToolCall(tc))
                }
                PartialPart::Thinking { content, signature } => {
                    let mut thinking = ThinkingPart::new(content);
                    thinking.signature = signature;
                    Some(ModelResponsePart::Thinking(thinking))
                }
                _ => None,
            })
            .collect();

        ModelResponse {
            parts,
            model_name: self.model_name,
            timestamp: self.timestamp,
            finish_reason: self.finish_reason,
            usage: self.usage,
            vendor_id: self.vendor_id,
            vendor_details: None,
            kind: "response".to_string(),
        }
    }

    /// Get a snapshot as a response (clones data).
    #[must_use]
    pub fn as_response(&self) -> ModelResponse {
        self.clone().finalize()
    }
}

/// Delta types for applying to partial response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseDelta {
    /// Text content delta.
    Text {
        /// Part index.
        index: usize,
        /// Text content.
        content: String,
    },
    /// Tool call delta.
    ToolCall {
        /// Part index.
        index: usize,
        /// Tool name (first delta only).
        name: Option<String>,
        /// Arguments delta.
        args: Option<String>,
        /// Tool call ID (first delta only).
        id: Option<String>,
    },
    /// Thinking delta.
    Thinking {
        /// Part index.
        index: usize,
        /// Thinking content.
        content: String,
        /// Signature (last delta only).
        signature: Option<String>,
    },
    /// Finish signal.
    Finish {
        /// Finish reason.
        reason: FinishReason,
    },
    /// Usage update.
    Usage {
        /// Current usage.
        usage: RequestUsage,
    },
}

impl PartialResponse {
    /// Apply a delta to the response.
    pub fn apply_delta(&mut self, delta: &ResponseDelta) {
        match delta {
            ResponseDelta::Text { index, content } => {
                self.apply_text_delta(*index, content);
            }
            ResponseDelta::ToolCall {
                index,
                name,
                args,
                id,
            } => {
                self.apply_tool_delta(*index, name.as_deref(), args.as_deref(), id.as_deref());
            }
            ResponseDelta::Thinking {
                index,
                content,
                signature,
            } => {
                self.apply_thinking_delta(*index, content, signature.as_deref());
            }
            ResponseDelta::Finish { reason } => {
                self.set_finish_reason(reason.clone());
            }
            ResponseDelta::Usage { usage } => {
                self.set_usage(usage.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_partial_response() {
        let pr = PartialResponse::new();
        assert!(pr.is_empty());
        assert_eq!(pr.num_parts(), 0);
    }

    #[test]
    fn test_text_accumulation() {
        let mut pr = PartialResponse::new();
        pr.apply_text_delta(0, "Hello, ");
        pr.apply_text_delta(0, "world!");

        assert_eq!(pr.text_content(), "Hello, world!");
        assert!(!pr.is_empty());
    }

    #[test]
    fn test_tool_call_accumulation() {
        let mut pr = PartialResponse::new();
        pr.apply_tool_delta(0, Some("search"), None, Some("call-1"));
        pr.apply_tool_delta(0, None, Some("{\"query\": "), None);
        pr.apply_tool_delta(0, None, Some("\"rust\"}"), None);

        let response = pr.finalize();
        assert_eq!(response.parts.len(), 1);

        if let ModelResponsePart::ToolCall(tc) = &response.parts[0] {
            assert_eq!(tc.tool_name, "search");
            assert_eq!(tc.tool_call_id, Some("call-1".to_string()));
        } else {
            panic!("Expected tool call part");
        }
    }

    #[test]
    fn test_thinking_accumulation() {
        let mut pr = PartialResponse::new();
        pr.apply_thinking_delta(0, "Let me think...", None);
        pr.apply_thinking_delta(0, " I need to", None);
        pr.apply_thinking_delta(0, " consider options.", Some("sig-123"));

        let response = pr.finalize();
        assert_eq!(response.parts.len(), 1);

        if let ModelResponsePart::Thinking(t) = &response.parts[0] {
            assert_eq!(t.content, "Let me think... I need to consider options.");
            assert_eq!(t.signature, Some("sig-123".to_string()));
        } else {
            panic!("Expected thinking part");
        }
    }

    #[test]
    fn test_multiple_parts() {
        let mut pr = PartialResponse::new();
        pr.apply_text_delta(0, "Hello");
        pr.apply_tool_delta(1, Some("search"), Some("{}"), None);
        pr.apply_text_delta(2, "World");

        let response = pr.finalize();
        assert_eq!(response.parts.len(), 3);
    }

    #[test]
    fn test_apply_delta() {
        let mut pr = PartialResponse::new();

        pr.apply_delta(&ResponseDelta::Text {
            index: 0,
            content: "Hello".to_string(),
        });

        pr.apply_delta(&ResponseDelta::Finish {
            reason: FinishReason::Stop,
        });

        let response = pr.finalize();
        assert_eq!(response.text_content(), "Hello");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_as_response_clones() {
        let mut pr = PartialResponse::new();
        pr.apply_text_delta(0, "Test");

        let snap1 = pr.as_response();
        pr.apply_text_delta(0, " more");
        let snap2 = pr.as_response();

        assert_eq!(snap1.text_content(), "Test");
        assert_eq!(snap2.text_content(), "Test more");
    }
}
