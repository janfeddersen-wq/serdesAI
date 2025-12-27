//! Ollama API types.

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
    /// Stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    /// Keep alive duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Response format ("json" for JSON mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role.
    pub role: String,
    /// Content.
    pub content: String,
    /// Images (base64 encoded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Model options.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Options {
    /// Temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top P.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top K.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Number of tokens to predict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Number of context tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    /// Repeat penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    /// Repeat last N.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
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
    pub description: String,
    /// Parameters.
    pub parameters: serde_json::Value,
}

/// Tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Function.
    pub function: FunctionCall,
}

/// Function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name.
    pub name: String,
    /// Arguments.
    pub arguments: serde_json::Value,
}

/// Chat response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    /// Model name.
    pub model: String,
    /// Creation time.
    pub created_at: String,
    /// Response message.
    pub message: ResponseMessage,
    /// Whether generation is done.
    pub done: bool,
    /// Reason for completion.
    pub done_reason: Option<String>,
    /// Total duration.
    pub total_duration: Option<u64>,
    /// Load duration.
    pub load_duration: Option<u64>,
    /// Prompt evaluation count.
    pub prompt_eval_count: Option<u32>,
    /// Prompt evaluation duration.
    pub prompt_eval_duration: Option<u64>,
    /// Evaluation count.
    pub eval_count: Option<u32>,
    /// Evaluation duration.
    pub eval_duration: Option<u64>,
}

/// Response message.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    /// Role.
    pub role: String,
    /// Content.
    pub content: String,
    /// Tool calls.
    pub tool_calls: Option<Vec<ToolCall>>,
}
