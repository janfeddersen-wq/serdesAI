//! Cohere API types.
//!
//! Cohere uses a unique API format with `message` + `chat_history`
//! rather than the OpenAI-style messages array.

use serde::{Deserialize, Serialize};

/// Chat request for Cohere v2 API.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    /// Model identifier.
    pub model: String,
    /// The current user message.
    pub message: String,
    /// Previous conversation history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_history: Option<Vec<ChatMessage>>,
    /// System preamble/prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preamble: Option<String>,
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f32>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Tool definitions for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Results from previous tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_results: Option<Vec<ToolResult>>,
}

impl ChatRequest {
    /// Create a new chat request.
    pub fn new(model: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            message: message.into(),
            chat_history: None,
            preamble: None,
            temperature: None,
            max_tokens: None,
            p: None,
            k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_results: None,
        }
    }
}

/// A message in chat history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: USER, CHATBOT, SYSTEM, or TOOL.
    pub role: Role,
    /// Message content.
    pub message: String,
    /// Tool calls made by the chatbot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    /// Create a user message.
    pub fn user(message: impl Into<String>) -> Self {
        Self { role: Role::User, message: message.into(), tool_calls: None }
    }

    /// Create a chatbot/assistant message.
    pub fn chatbot(message: impl Into<String>) -> Self {
        Self { role: Role::Chatbot, message: message.into(), tool_calls: None }
    }
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Role {
    /// User message.
    User,
    /// Assistant/chatbot message.
    Chatbot,
    /// System message.
    System,
    /// Tool result message.
    Tool,
}

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Parameter definitions as JSON schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_definitions: Option<serde_json::Value>,
}

/// Tool call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call.
    pub id: String,
    /// Name of the tool to call.
    pub name: String,
    /// Arguments as JSON.
    pub parameters: serde_json::Value,
}

/// Result from executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The ID of the tool call this result is for.
    pub call: ToolCallReference,
    /// The output from the tool.
    pub outputs: Vec<serde_json::Value>,
}

/// Reference to a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallReference {
    /// Tool call ID.
    pub id: String,
}

/// Chat response from Cohere.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    /// Response text.
    pub text: String,
    /// Generation ID.
    pub generation_id: Option<String>,
    /// Finish reason.
    pub finish_reason: Option<String>,
    /// Tool calls made by the model.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Token usage.
    pub meta: Option<ResponseMeta>,
}

/// Response metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMeta {
    /// API version.
    pub api_version: Option<ApiVersion>,
    /// Token usage.
    pub tokens: Option<TokenUsage>,
}

/// API version info.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiVersion {
    /// Version string.
    pub version: String,
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenUsage {
    /// Input tokens.
    pub input_tokens: Option<u32>,
    /// Output tokens.
    pub output_tokens: Option<u32>,
}

/// Streaming event from Cohere.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamEvent {
    /// Event type.
    pub event_type: String,
    /// Text delta (for text-generation events).
    pub text: Option<String>,
    /// Finish reason (for stream-end).
    pub finish_reason: Option<String>,
    /// Full response (for stream-end).
    pub response: Option<ChatResponse>,
    /// Tool calls (for tool-calls-generation).
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Cohere API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct CohereError {
    /// Error message.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialization() {
        let req = ChatRequest::new("command-r-plus", "Hello!");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("command-r-plus"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_chat_message() {
        let msg = ChatMessage::user("Hi there");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.message, "Hi there");
    }

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"USER\"");
        assert_eq!(serde_json::to_string(&Role::Chatbot).unwrap(), "\"CHATBOT\"");
    }
}
