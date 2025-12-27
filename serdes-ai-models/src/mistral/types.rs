//! Mistral API types.

use serde::{Deserialize, Serialize};

/// Chat request.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
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
    pub temperature: Option<f64>,
    /// Max tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Top P.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stream.
    pub stream: bool,
    /// Safe prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<i64>,
}

/// Message role.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message.
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool response.
    Tool,
}

/// Message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role.
    pub role: Role,
    /// Content.
    pub content: Content,
    /// Tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
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
        /// Text.
        text: String,
    },
    /// Image URL part.
    ImageUrl {
        /// Image URL.
        image_url: String,
    },
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Function.
    pub function: FunctionDef,
}

/// Function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Name.
    pub name: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameters.
    pub parameters: serde_json::Value,
}

/// Tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// ID.
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
    /// Name.
    pub name: String,
    /// Arguments.
    pub arguments: String,
}

/// Chat response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
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
    /// Usage.
    pub usage: Option<Usage>,
}

/// Response choice.
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    /// Index.
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
    pub role: Role,
    /// Content.
    pub content: Content,
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
