//! Groq API types.
//!
//! Groq uses an OpenAI-compatible API format.

use serde::{Deserialize, Serialize};

/// Chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    /// Model name.
    pub model: String,
    /// Messages.
    pub messages: Vec<Message>,
    /// Tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// Temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Max tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top P.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// User ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role.
    pub role: String,
    /// Content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (for tool role).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Text content.
    Text(String),
    /// Multi-part content.
    Parts(Vec<ContentPart>),
}

/// Content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text part.
    Text {
        /// Text content.
        text: String,
    },
    /// Image URL part.
    ImageUrl {
        /// Image URL.
        image_url: ImageUrl,
    },
}

/// Image URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL.
    pub url: String,
    /// Detail level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type (always "function").
    #[serde(rename = "type")]
    pub r#type: String,
    /// Function definition.
    pub function: FunctionDefinition,
}

/// Function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name.
    pub name: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameters schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call ID.
    pub id: String,
    /// Type.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Function.
    pub function: FunctionCall,
}

/// Function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments (JSON string).
    pub arguments: String,
}

/// Chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: u64,
    /// Model name.
    pub model: String,
    /// Choices.
    pub choices: Vec<Choice>,
    /// Usage statistics.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Response choice.
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    /// Choice index.
    pub index: u32,
    /// Message.
    pub message: ResponseMessage,
    /// Finish reason.
    pub finish_reason: Option<String>,
}

/// Response message.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    /// Role.
    pub role: String,
    /// Content.
    pub content: Option<String>,
    /// Tool calls.
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Usage statistics.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

/// Streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    /// Chunk ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: u64,
    /// Model name.
    pub model: String,
    /// Choices.
    pub choices: Vec<ChunkChoice>,
}

/// Streaming choice.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    /// Choice index.
    pub index: u32,
    /// Delta.
    pub delta: ChunkDelta,
    /// Finish reason.
    pub finish_reason: Option<String>,
}

/// Streaming delta.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ChunkDelta {
    /// Role.
    pub role: Option<String>,
    /// Content.
    pub content: Option<String>,
    /// Tool calls.
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// Streaming tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCall {
    /// Index.
    pub index: u32,
    /// Call ID.
    pub id: Option<String>,
    /// Type.
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    /// Function.
    pub function: Option<ChunkFunction>,
}

/// Streaming function.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkFunction {
    /// Function name.
    pub name: Option<String>,
    /// Arguments.
    pub arguments: Option<String>,
}
