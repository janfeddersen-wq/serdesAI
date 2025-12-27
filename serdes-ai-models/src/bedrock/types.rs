//! AWS Bedrock API types.
//!
//! Uses the Bedrock Converse API for unified access to all models.

use serde::{Deserialize, Serialize};

/// Converse API request.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseRequest {
    /// Model ID.
    pub model_id: String,
    /// Messages.
    pub messages: Vec<Message>,
    /// System prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemContent>>,
    /// Inference config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    /// Tool configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
}

/// System content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum SystemContent {
    /// Text system content.
    #[serde(rename = "text")]
    Text {
        /// Text.
        text: String,
    },
}

/// Message role.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User message.
    User,
    /// Assistant message.
    Assistant,
}

/// Message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role.
    pub role: Role,
    /// Content.
    pub content: Vec<Content>,
}

/// Content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum Content {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// Text.
        text: String,
    },
    /// Image content.
    #[serde(rename = "image")]
    Image {
        /// Image block.
        image: ImageBlock,
    },
    /// Tool use (from model).
    #[serde(rename = "toolUse")]
    ToolUse {
        /// Tool use ID.
        tool_use_id: String,
        /// Tool name.
        name: String,
        /// Tool input.
        input: serde_json::Value,
    },
    /// Tool result (to model).
    #[serde(rename = "toolResult")]
    ToolResult {
        /// Tool use ID.
        tool_use_id: String,
        /// Content.
        content: Vec<ToolResultContent>,
    },
    /// Thinking content (for Claude models via Bedrock).
    #[serde(rename = "reasoningContent")]
    ReasoningContent {
        /// Reasoning text.
        reasoning_text: Option<ReasoningText>,
        /// Redacted content (encrypted thinking).
        redacted_content: Option<String>,
    },
}

/// Reasoning text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningText {
    /// The reasoning text.
    pub text: String,
    /// Optional signature for verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Image block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBlock {
    /// Format (jpeg, png, gif, webp).
    pub format: String,
    /// Source.
    pub source: ImageSource,
}

/// Image source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum ImageSource {
    /// Base64 bytes.
    #[serde(rename = "bytes")]
    Bytes {
        /// Base64 encoded bytes.
        bytes: String,
    },
}

/// Tool result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum ToolResultContent {
    /// Text result.
    #[serde(rename = "text")]
    Text {
        /// Text.
        text: String,
    },
    /// JSON result.
    #[serde(rename = "json")]
    Json {
        /// JSON value.
        json: serde_json::Value,
    },
}

/// Inference configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    /// Max tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top P.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Tools.
    pub tools: Vec<Tool>,
    /// Tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceConfig>,
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum ToolChoiceConfig {
    /// Auto.
    #[serde(rename = "auto")]
    Auto,
    /// Any tool.
    #[serde(rename = "any")]
    Any,
    /// Specific tool.
    #[serde(rename = "tool")]
    Tool {
        /// Tool name.
        name: String,
    },
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Tool specification.
    pub tool_spec: ToolSpec,
}

/// Tool specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    /// Name.
    pub name: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Input schema.
    pub input_schema: ToolInputSchema,
}

/// Tool input schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// JSON schema.
    pub json: serde_json::Value,
}

/// Converse API response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseResponse {
    /// Output.
    pub output: Option<Output>,
    /// Stop reason.
    pub stop_reason: Option<String>,
    /// Usage.
    pub usage: Option<Usage>,
    /// Metrics.
    pub metrics: Option<Metrics>,
}

/// Response output.
#[derive(Debug, Clone, Deserialize)]
pub struct Output {
    /// Message.
    pub message: Option<Message>,
}

/// Usage statistics.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Input tokens.
    pub input_tokens: u32,
    /// Output tokens.
    pub output_tokens: u32,
    /// Total tokens.
    #[serde(default)]
    pub total_tokens: Option<u32>,
}

/// Response metrics.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    /// Latency in milliseconds.
    pub latency_ms: Option<u64>,
}
