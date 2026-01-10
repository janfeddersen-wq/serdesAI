//! Tool return types for tool execution results.
//!
//! This module defines the types used to represent the results of tool
//! executions that are sent back to the model.

use serde::{Deserialize, Serialize};

use super::content::ImageContent;

/// Content of a tool return.
///
/// Uses a tagged representation to avoid ambiguous deserialization when
/// multiple variants share compatible shapes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolReturnContent {
    /// Plain text result.
    Text {
        /// The text content.
        content: String,
    },
    /// JSON result.
    Json {
        /// The JSON content.
        content: serde_json::Value,
    },
    /// Image result.
    Image {
        /// The image content.
        image: ImageContent,
    },
    /// Error message.
    Error {
        /// The error payload.
        #[serde(flatten)]
        error: ToolReturnError,
    },
    /// Multiple return items.
    Multiple {
        /// The list of items.
        items: Vec<ToolReturnItem>,
    },
}

impl ToolReturnContent {
    /// Create text content.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { content: s.into() }
    }

    /// Create JSON content.
    #[must_use]
    pub fn json(value: serde_json::Value) -> Self {
        Self::Json { content: value }
    }

    /// Create error content.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            error: ToolReturnError::new(message),
        }
    }

    /// Create image content.
    #[must_use]
    pub fn image(image: ImageContent) -> Self {
        Self::Image { image }
    }

    /// Create multiple items.
    #[must_use]
    pub fn multiple(items: Vec<ToolReturnItem>) -> Self {
        Self::Multiple { items }
    }

    /// Check if this is an error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Create empty content.
    #[must_use]
    pub fn empty() -> Self {
        Self::Text {
            content: String::new(),
        }
    }

    /// Check if content is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text { content } => content.is_empty(),
            Self::Json { content } => content.is_null(),
            Self::Multiple { items } => items.is_empty(),
            _ => false,
        }
    }

    /// Get as text if this is text content.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { content } => Some(content),
            Self::Error { error } => Some(&error.message),
            _ => None,
        }
    }

    /// Get as JSON if this is JSON content.
    #[must_use]
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json { content } => Some(content),
            _ => None,
        }
    }

    /// Convert to string representation.
    #[must_use]
    pub fn to_string_content(&self) -> String {
        match self {
            Self::Text { content } => content.clone(),
            Self::Json { content } => serde_json::to_string(content).unwrap_or_default(),
            Self::Image { .. } => "[Image]".to_string(),
            Self::Error { error } => format!("Error: {}", error.message),
            Self::Multiple { items } => items
                .iter()
                .map(|i| i.to_string_content())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl Default for ToolReturnContent {
    fn default() -> Self {
        Self::Text {
            content: String::new(),
        }
    }
}

impl From<String> for ToolReturnContent {
    fn from(s: String) -> Self {
        Self::Text { content: s }
    }
}

impl From<&str> for ToolReturnContent {
    fn from(s: &str) -> Self {
        Self::Text {
            content: s.to_string(),
        }
    }
}

impl From<serde_json::Value> for ToolReturnContent {
    fn from(v: serde_json::Value) -> Self {
        Self::Json { content: v }
    }
}

/// Tool return error wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolReturnError {
    /// Error message.
    pub message: String,
    /// Error kind identifier.
    #[serde(default = "default_error_kind")]
    pub kind: String,
}

fn default_error_kind() -> String {
    "error".to_string()
}

impl ToolReturnError {
    /// Create a new error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "error".to_string(),
        }
    }
}

/// Individual item in a multi-item tool return.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolReturnItem {
    /// Text item.
    Text {
        /// The text content.
        content: String,
    },
    /// JSON item.
    Json {
        /// The JSON value.
        value: serde_json::Value,
    },
    /// Image item.
    Image {
        /// The image content.
        image: ImageContent,
    },
    /// Error item.
    Error {
        /// Error message.
        message: String,
    },
}

impl ToolReturnItem {
    /// Create text item.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text {
            content: content.into(),
        }
    }

    /// Create JSON item.
    #[must_use]
    pub fn json(value: serde_json::Value) -> Self {
        Self::Json { value }
    }

    /// Create image item.
    #[must_use]
    pub fn image(image: ImageContent) -> Self {
        Self::Image { image }
    }

    /// Create error item.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Convert to string representation.
    #[must_use]
    pub fn to_string_content(&self) -> String {
        match self {
            Self::Text { content } => content.clone(),
            Self::Json { value } => serde_json::to_string(value).unwrap_or_default(),
            Self::Image { .. } => "[Image]".to_string(),
            Self::Error { message } => format!("Error: {}", message),
        }
    }
}

/// Complete tool return with content and optional call ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolReturn {
    /// The return content.
    pub content: ToolReturnContent,
    /// Tool call ID this is responding to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ToolReturn {
    /// Create a new tool return.
    #[must_use]
    pub fn new(content: impl Into<ToolReturnContent>) -> Self {
        Self {
            content: content.into(),
            tool_call_id: None,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Create a successful text return.
    #[must_use]
    pub fn success(content: impl Into<String>) -> Self {
        Self::new(ToolReturnContent::text(content))
    }

    /// Create an error return.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(ToolReturnContent::error(message))
    }

    /// Create a JSON return.
    #[must_use]
    pub fn json(value: serde_json::Value) -> Self {
        Self::new(ToolReturnContent::json(value))
    }
}

impl Default for ToolReturn {
    fn default() -> Self {
        Self::new(ToolReturnContent::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_return_content_text() {
        let content = ToolReturnContent::text("Hello");
        assert!(!content.is_error());
        assert_eq!(content.to_string_content(), "Hello");
    }

    #[test]
    fn test_tool_return_content_error() {
        let content = ToolReturnContent::error("Something went wrong");
        assert!(content.is_error());
        assert!(content.to_string_content().contains("Error:"));
    }

    #[test]
    fn test_tool_return_content_json() {
        let content = ToolReturnContent::json(serde_json::json!({"status": "ok"}));
        let s = content.to_string_content();
        assert!(s.contains("status"));
    }

    #[test]
    fn test_tool_return_with_call_id() {
        let ret = ToolReturn::success("Done").with_tool_call_id("call_123");
        assert_eq!(ret.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_serde_roundtrip() {
        let ret = ToolReturn::json(serde_json::json!({"result": 42})).with_tool_call_id("id1");
        let json = serde_json::to_string(&ret).unwrap();
        let parsed: ToolReturn = serde_json::from_str(&json).unwrap();
        assert_eq!(ret, parsed);
    }
}
