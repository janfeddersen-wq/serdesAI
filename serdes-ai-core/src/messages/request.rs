//! Request message types for model interactions.
//!
//! This module defines the message types that are sent TO the model,
//! including system prompts, user prompts, tool returns, and retry prompts.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::content::UserContent;
use super::parts::BuiltinToolReturnPart;
use super::tool_return::ToolReturnContent;

/// A complete model request containing multiple parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    /// The request parts.
    pub parts: Vec<ModelRequestPart>,
    /// Kind identifier.
    #[serde(default = "default_request_kind")]
    pub kind: String,
}

fn default_request_kind() -> String {
    "request".to_string()
}

impl ModelRequest {
    /// Create a new empty request.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            kind: "request".to_string(),
        }
    }

    /// Create a request with the given parts.
    #[must_use]
    pub fn with_parts(parts: Vec<ModelRequestPart>) -> Self {
        Self {
            parts,
            kind: "request".to_string(),
        }
    }

    /// Add a part.
    pub fn add_part(&mut self, part: ModelRequestPart) {
        self.parts.push(part);
    }

    /// Add a system prompt.
    pub fn add_system_prompt(&mut self, content: impl Into<String>) {
        self.parts.push(ModelRequestPart::SystemPrompt(
            SystemPromptPart::new(content),
        ));
    }

    /// Add a user prompt.
    pub fn add_user_prompt(&mut self, content: impl Into<UserContent>) {
        self.parts
            .push(ModelRequestPart::UserPrompt(UserPromptPart::new(content)));
    }

    /// Get all system prompts.
    #[must_use]
    pub fn system_prompts(&self) -> impl Iterator<Item = &SystemPromptPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelRequestPart::SystemPrompt(s) => Some(s),
            _ => None,
        })
    }

    /// Get all user prompts.
    #[must_use]
    pub fn user_prompts(&self) -> impl Iterator<Item = &UserPromptPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelRequestPart::UserPrompt(u) => Some(u),
            _ => None,
        })
    }

    /// Get all tool returns.
    #[must_use]
    pub fn tool_returns(&self) -> impl Iterator<Item = &ToolReturnPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelRequestPart::ToolReturn(t) => Some(t),
            _ => None,
        })
    }

    /// Get all builtin tool returns.
    #[must_use]
    pub fn builtin_tool_returns(&self) -> impl Iterator<Item = &BuiltinToolReturnPart> {
        self.parts.iter().filter_map(|p| match p {
            ModelRequestPart::BuiltinToolReturn(b) => Some(b),
            _ => None,
        })
    }

    /// Get all system prompts as a vector.
    #[deprecated(note = "Use system_prompts() iterator instead")]
    pub fn system_prompts_vec(&self) -> Vec<&SystemPromptPart> {
        self.system_prompts().collect()
    }

    /// Get all user prompts as a vector.
    #[deprecated(note = "Use user_prompts() iterator instead")]
    pub fn user_prompts_vec(&self) -> Vec<&UserPromptPart> {
        self.user_prompts().collect()
    }

    /// Get all tool returns as a vector.
    #[deprecated(note = "Use tool_returns() iterator instead")]
    pub fn tool_returns_vec(&self) -> Vec<&ToolReturnPart> {
        self.tool_returns().collect()
    }

    /// Get all builtin tool returns as a vector.
    #[deprecated(note = "Use builtin_tool_returns() iterator instead")]
    pub fn builtin_tool_returns_vec(&self) -> Vec<&BuiltinToolReturnPart> {
        self.builtin_tool_returns().collect()
    }

    /// Add a builtin tool return.
    pub fn add_builtin_tool_return(&mut self, part: BuiltinToolReturnPart) {
        self.parts.push(ModelRequestPart::BuiltinToolReturn(part));
    }

    /// Check if the request is empty.
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

impl Default for ModelRequest {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<ModelRequestPart> for ModelRequest {
    fn from_iter<T: IntoIterator<Item = ModelRequestPart>>(iter: T) -> Self {
        Self::with_parts(iter.into_iter().collect())
    }
}

/// Individual parts of a model request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "part_kind", rename_all = "kebab-case")]
pub enum ModelRequestPart {
    /// System prompt.
    SystemPrompt(SystemPromptPart),
    /// User prompt.
    UserPrompt(UserPromptPart),
    /// Tool return.
    ToolReturn(ToolReturnPart),
    /// Retry prompt.
    RetryPrompt(RetryPromptPart),
    /// Builtin tool return (web search results, code execution output, etc.).
    BuiltinToolReturn(BuiltinToolReturnPart),
}

impl ModelRequestPart {
    /// Get the timestamp of this part.
    #[must_use]
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::SystemPrompt(p) => p.timestamp,
            Self::UserPrompt(p) => p.timestamp,
            Self::ToolReturn(p) => p.timestamp,
            Self::RetryPrompt(p) => p.timestamp,
            Self::BuiltinToolReturn(p) => p.timestamp,
        }
    }

    /// Get the part kind string.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        match self {
            Self::SystemPrompt(_) => SystemPromptPart::PART_KIND,
            Self::UserPrompt(_) => UserPromptPart::PART_KIND,
            Self::ToolReturn(_) => ToolReturnPart::PART_KIND,
            Self::RetryPrompt(_) => RetryPromptPart::PART_KIND,
            Self::BuiltinToolReturn(_) => BuiltinToolReturnPart::PART_KIND,
        }
    }

    /// Check if this is a builtin tool return.
    #[must_use]
    pub fn is_builtin_tool_return(&self) -> bool {
        matches!(self, Self::BuiltinToolReturn(_))
    }
}

/// System prompt part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemPromptPart {
    /// The system prompt content.
    pub content: String,
    /// When this part was created.
    pub timestamp: DateTime<Utc>,
    /// Reference to a dynamic prompt source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_ref: Option<String>,
}

impl SystemPromptPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "system-prompt";

    /// Create a new system prompt part.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            timestamp: Utc::now(),
            dynamic_ref: None,
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the dynamic reference.
    #[must_use]
    pub fn with_dynamic_ref(mut self, ref_name: impl Into<String>) -> Self {
        self.dynamic_ref = Some(ref_name.into());
        self
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }
}

impl From<String> for SystemPromptPart {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for SystemPromptPart {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// User prompt part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserPromptPart {
    /// The user prompt content.
    pub content: UserContent,
    /// When this part was created.
    pub timestamp: DateTime<Utc>,
}

impl UserPromptPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "user-prompt";

    /// Create a new user prompt part.
    #[must_use]
    pub fn new(content: impl Into<UserContent>) -> Self {
        Self {
            content: content.into(),
            timestamp: Utc::now(),
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Get content as text if it's text content.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        self.content.as_text()
    }
}

impl From<String> for UserPromptPart {
    fn from(s: String) -> Self {
        Self::new(UserContent::text(s))
    }
}

impl From<&str> for UserPromptPart {
    fn from(s: &str) -> Self {
        Self::new(UserContent::text(s))
    }
}

/// Tool return part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolReturnPart {
    /// Name of the tool.
    pub tool_name: String,
    /// The return content.
    pub content: ToolReturnContent,
    /// Tool call ID this is responding to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// When this part was created.
    pub timestamp: DateTime<Utc>,
}

impl ToolReturnPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "tool-return";

    /// Create a new tool return part.
    #[must_use]
    pub fn new(tool_name: impl Into<String>, content: impl Into<ToolReturnContent>) -> Self {
        Self {
            tool_name: tool_name.into(),
            content: content.into(),
            tool_call_id: None,
            timestamp: Utc::now(),
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Create a success return.
    #[must_use]
    pub fn success(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(tool_name, ToolReturnContent::text(content))
    }

    /// Create an error return.
    #[must_use]
    pub fn error(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(tool_name, ToolReturnContent::error(message))
    }
}

/// Retry content - either text or structured error info.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RetryContent {
    /// Plain text retry message.
    Text(String),
    /// Structured retry info.
    Structured {
        /// The error message.
        message: String,
        /// Optional validation errors.
        #[serde(skip_serializing_if = "Option::is_none")]
        errors: Option<Vec<String>>,
    },
}

impl RetryContent {
    /// Create text retry content.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create structured retry content.
    #[must_use]
    pub fn structured(message: impl Into<String>, errors: Option<Vec<String>>) -> Self {
        Self::Structured {
            message: message.into(),
            errors,
        }
    }

    /// Get the message.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::Structured { message, .. } => message,
        }
    }
}

impl Default for RetryContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for RetryContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for RetryContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

/// Retry prompt part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryPromptPart {
    /// The retry content.
    pub content: RetryContent,
    /// Tool name if this is a tool retry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Tool call ID if this is a tool retry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// When this part was created.
    pub timestamp: DateTime<Utc>,
}

impl RetryPromptPart {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "retry-prompt";

    /// Create a new retry prompt part.
    #[must_use]
    pub fn new(content: impl Into<RetryContent>) -> Self {
        Self {
            content: content.into(),
            tool_name: None,
            tool_call_id: None,
            timestamp: Utc::now(),
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the tool name.
    #[must_use]
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Create a tool retry.
    #[must_use]
    pub fn tool_retry(
        tool_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(message.into()).with_tool_name(tool_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_request_new() {
        let mut req = ModelRequest::new();
        assert!(req.is_empty());
        
        req.add_system_prompt("You are a helpful assistant.");
        req.add_user_prompt("Hello!");
        
        assert_eq!(req.len(), 2);
        assert_eq!(req.system_prompts().count(), 1);
        assert_eq!(req.user_prompts().count(), 1);
    }

    #[test]
    fn test_system_prompt_part() {
        let part = SystemPromptPart::new("Be helpful")
            .with_dynamic_ref("main_prompt");
        assert_eq!(part.content, "Be helpful");
        assert_eq!(part.dynamic_ref, Some("main_prompt".to_string()));
        assert_eq!(part.part_kind(), "system-prompt");
    }

    #[test]
    fn test_tool_return_part() {
        let part = ToolReturnPart::success("get_weather", "72°F, sunny")
            .with_tool_call_id("call_123");
        assert_eq!(part.tool_name, "get_weather");
        assert_eq!(part.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_retry_prompt_part() {
        let part = RetryPromptPart::tool_retry("my_tool", "Invalid JSON")
            .with_tool_call_id("id1");
        assert_eq!(part.tool_name, Some("my_tool".to_string()));
        assert_eq!(part.content.message(), "Invalid JSON");
    }

    #[test]
    fn test_serde_roundtrip() {
        let req = ModelRequest::with_parts(vec![
            ModelRequestPart::SystemPrompt(SystemPromptPart::new("System")),
            ModelRequestPart::UserPrompt(UserPromptPart::new("User")),
        ]);
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ModelRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req.len(), parsed.len());
    }

    #[test]
    fn test_builtin_tool_return() {
        use crate::messages::parts::{BuiltinToolReturnContent, WebSearchResults, WebSearchResult};

        let results = WebSearchResults::new(
            "rust programming",
            vec![WebSearchResult::new("Rust", "https://rust-lang.org")],
        );
        let content = BuiltinToolReturnContent::web_search(results);
        let part = BuiltinToolReturnPart::new("web_search", content, "call_123");

        let mut req = ModelRequest::new();
        req.add_builtin_tool_return(part);

        assert_eq!(req.len(), 1);
        assert_eq!(req.builtin_tool_returns().count(), 1);

        let returns: Vec<_> = req.builtin_tool_returns().collect();
        assert_eq!(returns[0].tool_name, "web_search");
        assert_eq!(returns[0].tool_call_id, "call_123");
    }

    #[test]
    fn test_model_request_part_is_builtin_tool_return() {
        use crate::messages::parts::{BuiltinToolReturnContent, CodeExecutionResult};

        let result = CodeExecutionResult::new("print(1)").with_stdout("1\n");
        let content = BuiltinToolReturnContent::code_execution(result);
        let part = BuiltinToolReturnPart::new("code_execution", content, "call_456");
        let request_part = ModelRequestPart::BuiltinToolReturn(part);

        assert!(request_part.is_builtin_tool_return());
        assert_eq!(request_part.part_kind(), "builtin-tool-return");
    }

    #[test]
    fn test_serde_roundtrip_with_builtin_tool_return() {
        use crate::messages::parts::{BuiltinToolReturnContent, FileSearchResults, FileSearchResult};

        let results = FileSearchResults::new(
            "main function",
            vec![FileSearchResult::new("main.rs", "fn main() {}")],
        );
        let content = BuiltinToolReturnContent::file_search(results);
        let part = BuiltinToolReturnPart::new("file_search", content, "call_789");

        let req = ModelRequest::with_parts(vec![
            ModelRequestPart::UserPrompt(UserPromptPart::new("Search files")),
            ModelRequestPart::BuiltinToolReturn(part),
        ]);

        let json = serde_json::to_string(&req).unwrap();
        let parsed: ModelRequest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(req.len(), parsed.len());
        assert_eq!(parsed.builtin_tool_returns().count(), 1);
    }
}
