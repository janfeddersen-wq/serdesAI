//! ChatGPT OAuth types.
//!
//! These are API request/response types for the ChatGPT OAuth flow.

#![allow(missing_docs)] // DTO fields are self-documenting

use serde::{Deserialize, Serialize};

/// ChatGPT Codex API configuration.
#[derive(Debug, Clone)]
pub struct ChatGptConfig {
    /// Base URL for the Codex API
    pub api_base_url: String,
    /// Model prefix for display
    pub prefix: String,
    /// Default context length
    pub context_length: usize,
}

impl Default for ChatGptConfig {
    fn default() -> Self {
        Self {
            api_base_url: "https://chatgpt.com/backend-api/codex".to_string(),
            prefix: "chatgpt-".to_string(),
            context_length: 272000,
        }
    }
}

/// Request body for ChatGPT Codex Responses API.
#[derive(Debug, Serialize)]
pub struct CodexRequest {
    pub model: String,
    /// System instructions (REQUIRED by Responses API)
    pub instructions: String,
    /// User input - list of messages and function outputs
    pub input: Vec<InputItem>,
    /// Required by ChatGPT Codex API - must be false
    #[serde(default)]
    pub store: bool,
    /// Required by ChatGPT Codex API - must be true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Reasoning settings for GPT-5 and o-series models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
}

/// Reasoning configuration for GPT-5 and o-series models
#[derive(Debug, Serialize, Clone)]
pub struct ReasoningConfig {
    pub effort: String,
    pub summary: String,
}

/// Function call from assistant for Responses API
#[derive(Debug, Serialize, Clone)]
pub struct FunctionCallItem {
    #[serde(rename = "type")]
    pub call_type: String,  // Always "function_call"
    pub name: String,
    pub arguments: String,
    pub call_id: String,
}

/// Function call output for Responses API (tool return)
#[derive(Debug, Serialize, Clone)]
pub struct FunctionCallOutput {
    #[serde(rename = "type")]
    pub output_type: String,  // Always "function_call_output"
    pub call_id: String,
    pub output: String,
}

/// Input item for Responses API - can be a message, function call, or function output
#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum InputItem {
    Message(CodexMessage),
    FunctionCall(FunctionCallItem),
    FunctionOutput(FunctionCallOutput),
}

/// Message in a Codex request.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CodexMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message content (string or parts array).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Content part for multi-modal messages.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Tool call in assistant message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Response from Codex API.
#[derive(Debug, Deserialize)]
pub struct CodexResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}
