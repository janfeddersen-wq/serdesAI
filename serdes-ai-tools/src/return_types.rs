//! Tool return types.
//!
//! This module provides types for tool execution results, including
//! text, JSON, images, and error returns.

use serde::{Deserialize, Serialize};
use serdes_ai_core::messages::{ImageContent, ToolReturnContent};

/// What a tool returns after execution.
///
/// This wraps the content returned by a tool along with optional
/// metadata like the tool call ID.
#[derive(Debug, Clone)]
pub struct ToolReturn {
    /// The content returned by the tool.
    pub content: ToolReturnContent,
    /// The tool call ID this is responding to.
    pub tool_call_id: Option<String>,
}

impl ToolReturn {
    /// Create a text return.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content: ToolReturnContent::text(s),
            tool_call_id: None,
        }
    }

    /// Create a JSON return.
    #[must_use]
    pub fn json(value: serde_json::Value) -> Self {
        Self {
            content: ToolReturnContent::json(value),
            tool_call_id: None,
        }
    }

    /// Create a JSON return from a serializable value.
    pub fn from_value<T: Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        let json = serde_json::to_value(value)?;
        Ok(Self::json(json))
    }

    /// Create an image return.
    #[must_use]
    pub fn image(image: ImageContent) -> Self {
        Self {
            content: ToolReturnContent::image(image),
            tool_call_id: None,
        }
    }

    /// Create an error return.
    #[must_use]
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            content: ToolReturnContent::error(msg),
            tool_call_id: None,
        }
    }

    /// Create an empty return.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            content: ToolReturnContent::empty(),
            tool_call_id: None,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Get the tool call ID.
    #[must_use]
    pub fn call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    /// Check if this is an error return.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.content.is_error()
    }

    /// Check if this is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the content as text if applicable.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        self.content.as_text()
    }

    /// Get the content as JSON if applicable.
    #[must_use]
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        self.content.as_json()
    }

    /// Convert to the core ToolReturnContent.
    #[must_use]
    pub fn into_content(self) -> ToolReturnContent {
        self.content
    }
}

impl Default for ToolReturn {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<String> for ToolReturn {
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

impl From<&str> for ToolReturn {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

impl From<serde_json::Value> for ToolReturn {
    fn from(v: serde_json::Value) -> Self {
        Self::json(v)
    }
}

impl From<ToolReturnContent> for ToolReturn {
    fn from(content: ToolReturnContent) -> Self {
        Self {
            content,
            tool_call_id: None,
        }
    }
}

impl From<ImageContent> for ToolReturn {
    fn from(image: ImageContent) -> Self {
        Self::image(image)
    }
}

/// Result of a tool execution.
pub type ToolResult = Result<ToolReturn, crate::ToolError>;

/// Trait for types that can be converted to a tool return.
pub trait IntoToolReturn {
    /// Convert to a tool return.
    fn into_tool_return(self) -> ToolReturn;
}

impl IntoToolReturn for ToolReturn {
    fn into_tool_return(self) -> ToolReturn {
        self
    }
}

impl IntoToolReturn for String {
    fn into_tool_return(self) -> ToolReturn {
        ToolReturn::text(self)
    }
}

impl IntoToolReturn for &str {
    fn into_tool_return(self) -> ToolReturn {
        ToolReturn::text(self)
    }
}

impl IntoToolReturn for serde_json::Value {
    fn into_tool_return(self) -> ToolReturn {
        ToolReturn::json(self)
    }
}

impl IntoToolReturn for () {
    fn into_tool_return(self) -> ToolReturn {
        ToolReturn::empty()
    }
}

impl<T: Serialize> IntoToolReturn for Vec<T> {
    fn into_tool_return(self) -> ToolReturn {
        match serde_json::to_value(&self) {
            Ok(v) => ToolReturn::json(v),
            Err(e) => ToolReturn::error(format!("Failed to serialize: {e}")),
        }
    }
}

/// Serializable tool result for storage/transmission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableToolResult {
    /// Whether the call succeeded.
    pub success: bool,
    /// The return content (if successful).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    /// Error message (if failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl SerializableToolResult {
    /// Create a success result.
    #[must_use]
    pub fn success(content: serde_json::Value) -> Self {
        Self {
            success: true,
            content: Some(content),
            error: None,
            tool_call_id: None,
        }
    }

    /// Create a failure result.
    #[must_use]
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            content: None,
            error: Some(error.into()),
            tool_call_id: None,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

impl From<&ToolReturn> for SerializableToolResult {
    fn from(ret: &ToolReturn) -> Self {
        let content = if ret.is_error() {
            None
        } else {
            Some(serde_json::to_value(&ret.content).unwrap_or_default())
        };

        Self {
            success: !ret.is_error(),
            content,
            error: if ret.is_error() {
                ret.as_text().map(String::from)
            } else {
                None
            },
            tool_call_id: ret.tool_call_id.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_return_text() {
        let ret = ToolReturn::text("Hello");
        assert!(!ret.is_error());
        assert_eq!(ret.as_text(), Some("Hello"));
    }

    #[test]
    fn test_tool_return_json() {
        let ret = ToolReturn::json(serde_json::json!({"x": 1}));
        assert!(!ret.is_error());
        assert!(ret.as_json().is_some());
    }

    #[test]
    fn test_tool_return_error() {
        let ret = ToolReturn::error("Something went wrong");
        assert!(ret.is_error());
    }

    #[test]
    fn test_tool_return_with_call_id() {
        let ret = ToolReturn::text("Result").with_call_id("call_123");
        assert_eq!(ret.call_id(), Some("call_123"));
    }

    #[test]
    fn test_from_string() {
        let ret: ToolReturn = "test".into();
        assert_eq!(ret.as_text(), Some("test"));
    }

    #[test]
    fn test_from_json_value() {
        let ret: ToolReturn = serde_json::json!({"a": 1}).into();
        assert!(ret.as_json().is_some());
    }

    #[test]
    fn test_from_value() {
        #[derive(Serialize)]
        struct Data {
            x: i32,
        }
        let data = Data { x: 42 };
        let ret = ToolReturn::from_value(&data).unwrap();
        assert_eq!(ret.as_json().unwrap()["x"], 42);
    }

    #[test]
    fn test_into_tool_return_trait() {
        let ret1 = "hello".into_tool_return();
        assert_eq!(ret1.as_text(), Some("hello"));

        let ret2 = ().into_tool_return();
        assert!(ret2.is_empty());
    }

    #[test]
    fn test_serializable_result() {
        let ret = ToolReturn::text("OK").with_call_id("id1");
        let serializable = SerializableToolResult::from(&ret);
        assert!(serializable.success);
        assert_eq!(serializable.tool_call_id, Some("id1".to_string()));
    }
}
