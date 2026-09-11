//! OpenAI Responses API implementation.
//!
//! The Responses API is OpenAI's new API that supports native reasoning
//! models like o1, o3, and gpt-5 with built-in tool execution.
//!
//! Key differences from Chat Completions:
//! - Uses `input` instead of `messages`
//! - Has native reasoning support with configurable effort
//! - Supports built-in tools (web search, code interpreter, file search, etc.)
//! - Different output format with `ResponseOutputItem` variants

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::{ModelProfile, openai_o1_profile};
use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serdes_ai_core::messages::{
    ImageContent, RetryPromptPart, TextPart, ThinkingPart, ToolCallArgs, ToolCallPart,
    ToolReturnPart, UserContent, UserContentPart, UserPromptPart,
};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage,
};
use serdes_ai_tools::ToolDefinition;
use std::time::Duration;

// ============================================================================
// Responses API Settings
// ============================================================================

/// OpenAI Responses API model settings.
#[derive(Debug, Clone, Default)]
pub struct OpenAIResponsesModelSettings {
    /// Reasoning effort (e.g. "low", "high", "xhigh", "max", or any custom string)
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Reasoning summary: "concise", "detailed", "auto"
    pub reasoning_summary: Option<ReasoningSummary>,

    /// Whether to send reasoning IDs back to the API (for continuation)
    pub send_reasoning_ids: bool,

    /// Include log probabilities
    pub logprobs: Option<bool>,

    /// Top logprobs count
    pub top_logprobs: Option<u32>,

    /// Service tier
    pub service_tier: Option<ServiceTier>,

    /// Truncation mode
    pub truncation: Option<TruncationMode>,

    /// Previous response ID for continuation
    pub previous_response_id: Option<String>,
}

/// Reasoning effort level for reasoning models.
///
/// Known efforts are named variants; newer models may accept efforts this
/// crate does not know about, which can be expressed with
/// [`ReasoningEffort::Custom`]. The value is sent to the API verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReasoningEffort {
    /// Minimal reasoning - fastest responses (gpt-5 family)
    Minimal,
    /// Low reasoning
    Low,
    /// Balanced reasoning - default
    #[default]
    Medium,
    /// Deep reasoning
    High,
    /// Extra-deep reasoning (newer models, e.g. gpt-5.1)
    XHigh,
    /// Maximum reasoning (newer models, e.g. gpt-5.1-pro)
    Max,
    /// Any other effort string, passed through verbatim.
    Custom(String),
}

impl ReasoningEffort {
    /// The effort string sent to the API.
    fn as_str(&self) -> &str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            Self::Custom(s) => s,
        }
    }

    /// Parse an effort string. Known (case-insensitive) values map to named
    /// variants; anything else becomes [`ReasoningEffort::Custom`] with the
    /// original string preserved.
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            _ => Self::Custom(s.to_string()),
        }
    }
}

impl From<&str> for ReasoningEffort {
    fn from(s: &str) -> Self {
        Self::parse(s)
    }
}

impl From<String> for ReasoningEffort {
    fn from(s: String) -> Self {
        Self::parse(&s)
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for ReasoningEffort {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Reasoning summary format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningSummary {
    /// Concise summary of reasoning
    Concise,
    /// Detailed summary of reasoning
    Detailed,
    /// Let the model decide
    #[default]
    Auto,
}

impl ReasoningSummary {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Concise => "concise",
            Self::Detailed => "detailed",
            Self::Auto => "auto",
        }
    }
}

impl Serialize for ReasoningSummary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Service tier for API requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServiceTier {
    /// Automatic tier selection
    #[default]
    Auto,
    /// Default tier
    Default,
    /// Flexible tier (may have variable latency)
    Flex,
    /// Priority tier (higher availability)
    Priority,
}

impl Serialize for ServiceTier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
            Self::Priority => "priority",
        };
        serializer.serialize_str(s)
    }
}

/// Truncation mode for input that exceeds context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationMode {
    /// Disable truncation (error on overflow)
    #[default]
    Disabled,
    /// Auto-truncate older messages
    Auto,
}

impl Serialize for TruncationMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            Self::Disabled => "disabled",
            Self::Auto => "auto",
        };
        serializer.serialize_str(s)
    }
}

// ============================================================================
// Responses API Request Types
// ============================================================================

/// Request body for the Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesApiRequest {
    /// Model to use.
    pub model: String,
    /// Request encrypted reasoning for stateless replay.
    pub include: Vec<String>,
    /// Input messages/content.
    pub input: Vec<ResponseInput>,
    /// System instructions (replaces system message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,
    /// Reasoning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Whether to stream the response.
    pub stream: bool,
    /// Previous response ID for multi-turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Truncation settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationConfig>,
    /// User identifier for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Store response for later retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Metadata for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonValue>,
}

/// Reasoning configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ReasoningConfig {
    /// Reasoning effort level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Summary format for reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

/// Truncation configuration.
#[derive(Debug, Clone, Serialize)]
pub struct TruncationConfig {
    /// Truncation type.
    #[serde(rename = "type")]
    pub truncation_type: TruncationMode,
}

/// Input item for the Responses API.
#[derive(Clone, Serialize)]
#[serde(tag = "role")]
#[allow(missing_docs)]
pub enum ResponseInput {
    /// User message.
    #[serde(rename = "user")]
    User { content: ResponseInputContent },
    /// Assistant message (for multi-turn).
    #[serde(rename = "assistant")]
    Assistant {
        content: ResponseInputContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_id: Option<String>,
    },
    /// Tool output.
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
    /// Native Responses item (reasoning, function call, or function output).
    #[serde(untagged)]
    Item(JsonValue),
}

impl std::fmt::Debug for ResponseInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ResponseInput(<redacted>)")
    }
}

/// Content for response input.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ResponseInputContent {
    /// Simple text.
    Text(String),
    /// Multi-part content.
    Parts(Vec<ResponseInputPart>),
}

/// Part of multi-part input content.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ResponseInputPart {
    /// Text content.
    #[serde(rename = "input_text")]
    Text { text: String },
    /// Image from URL.
    #[serde(rename = "input_image")]
    ImageUrl {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Image from base64.
    #[serde(rename = "input_image")]
    ImageBase64 {
        image_url: String, // data:mime;base64,xxx format
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Audio input.
    #[serde(rename = "input_audio")]
    Audio { data: String, format: String },
}

// ============================================================================
// Responses API Tool Types
// ============================================================================

/// Tool definition for Responses API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ResponseTool {
    /// Function tool (custom functions).
    #[serde(rename = "function")]
    Function {
        name: String,
        description: String,
        parameters: JsonValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// Web search built-in tool.
    #[serde(rename = "web_search_preview")]
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    /// Code interpreter built-in tool.
    #[serde(rename = "code_interpreter")]
    CodeInterpreter {
        #[serde(skip_serializing_if = "Option::is_none")]
        container: Option<ContainerConfig>,
    },
    /// File search built-in tool.
    #[serde(rename = "file_search")]
    FileSearch {
        vector_store_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<RankingOptions>,
    },
    /// Image generation built-in tool.
    #[serde(rename = "image_generation")]
    ImageGeneration {
        #[serde(skip_serializing_if = "Option::is_none")]
        background: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_image_mask: Option<ImageMask>,
        #[serde(skip_serializing_if = "Option::is_none")]
        moderation: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_compression: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        partial_images: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        quality: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<String>,
    },
    /// MCP (Model Context Protocol) tool.
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        server_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_tools: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        headers: Option<JsonValue>,
    },
}

/// User location for web search.
#[derive(Debug, Clone, Serialize)]
#[allow(missing_docs)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// Container configuration for code interpreter.
#[derive(Debug, Clone, Serialize)]
#[allow(missing_docs)]
pub struct ContainerConfig {
    #[serde(rename = "type")]
    pub container_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,
}

/// Ranking options for file search.
#[derive(Debug, Clone, Serialize)]
#[allow(missing_docs)]
pub struct RankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

/// Image mask for image generation.
#[derive(Debug, Clone, Serialize)]
#[allow(missing_docs)]
pub struct ImageMask {
    pub image_url: String,
    #[serde(rename = "type")]
    pub mask_type: String,
}

// ============================================================================
// Responses API Response Types
// ============================================================================

/// Response from the Responses API.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesApiResponse {
    /// Response ID.
    pub id: String,
    /// Object type (always "response").
    pub object: String,
    /// Creation timestamp.
    pub created_at: u64,
    /// Model used.
    pub model: String,
    /// Output items.
    pub output: Vec<ResponseOutputItem>,
    /// Token usage.
    pub usage: Option<ResponseUsage>,
    /// Response status.
    pub status: ResponseStatus,
    /// Provider reason for a partial response.
    pub incomplete_details: Option<JsonValue>,
    /// Error if any.
    pub error: Option<ResponseError>,
    /// Metadata.
    pub metadata: Option<JsonValue>,
    /// Service tier used.
    pub service_tier: Option<String>,
}

/// Response status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response is complete.
    Completed,
    /// Response failed.
    Failed,
    /// Response was cancelled.
    Cancelled,
    /// Response is incomplete (truncated).
    Incomplete,
    /// Response is in progress (streaming).
    InProgress,
}

/// Error in response.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

/// Output item from the Responses API.
#[derive(Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ResponseOutputItem {
    /// Reasoning/thinking output.
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        encrypted_content: Option<String>,
        #[serde(default)]
        summary: Vec<ReasoningSummaryItem>,
        status: Option<String>,
    },
    /// Text message output.
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<MessageContentItem>,
        status: Option<String>,
    },
    /// Function tool call.
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: Option<String>,
    },
    /// Function tool call output.
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    /// Web search tool call.
    #[serde(rename = "web_search_call")]
    WebSearchCall { id: String, status: Option<String> },
    /// Code interpreter tool call.
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        id: String,
        code: Option<String>,
        #[serde(default)]
        results: Vec<CodeInterpreterResult>,
        status: Option<String>,
    },
    /// File search tool call.
    #[serde(rename = "file_search_call")]
    FileSearchCall {
        id: String,
        #[serde(default)]
        results: Vec<FileSearchResult>,
        status: Option<String>,
    },
    /// Image generation tool call.
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall {
        id: String,
        result: Option<ImageGenerationResult>,
        status: Option<String>,
    },
    /// MCP tool call.
    #[serde(rename = "mcp_call")]
    McpCall {
        id: String,
        server_label: String,
        tool_name: String,
        arguments: String,
        output: Option<String>,
        error: Option<String>,
        status: Option<String>,
    },
}

/// Reasoning summary item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ReasoningSummaryItem {
    #[serde(rename = "summary_text")]
    Text { text: String },
}

impl std::fmt::Debug for ResponseOutputItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ResponseOutputItem(<redacted>)")
    }
}

/// Message content item.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum MessageContentItem {
    #[serde(rename = "output_text")]
    Text {
        text: String,
        #[serde(default)]
        annotations: Vec<JsonValue>,
    },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

/// Code interpreter result.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum CodeInterpreterResult {
    #[serde(rename = "logs")]
    Logs { logs: String },
    #[serde(rename = "image")]
    Image { image_url: String },
    #[serde(rename = "file")]
    File { file_id: String, filename: String },
}

/// File search result.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct FileSearchResult {
    pub file_id: String,
    pub filename: String,
    pub score: Option<f64>,
    pub text: Option<String>,
}

/// Image generation result.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct ImageGenerationResult {
    pub image: Option<String>, // base64 or URL
    pub revised_prompt: Option<String>,
}

/// Token usage for Responses API.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Input token details.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct InputTokensDetails {
    pub cached_tokens: Option<u64>,
}

/// Output token details.
#[derive(Debug, Clone, Deserialize)]
#[allow(missing_docs)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: Option<u64>,
}

// ============================================================================
// OpenAI Responses Model
// ============================================================================

/// OpenAI Responses API model.
///
/// This model implements the new Responses API which supports native reasoning
/// models like o1, o3, and future gpt-5 variants.
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai_models::openai::{OpenAIResponsesModel, ReasoningEffort};
///
/// let model = OpenAIResponsesModel::from_env("o3-mini")?
///     .with_reasoning_effort(ReasoningEffort::High);
///
/// let response = model.request(&messages, &settings, &params).await?;
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIResponsesModel {
    model_name: String,
    client: Client,
    api_key: String,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    profile: ModelProfile,
    default_timeout: Duration,
    default_settings: OpenAIResponsesModelSettings,
}

impl OpenAIResponsesModel {
    /// Create a new OpenAI Responses model.
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let profile = Self::profile_for_model(&model_name);

        Self {
            model_name,
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            project: None,
            profile,
            default_timeout: Duration::from_secs(300), // Longer for reasoning
            default_settings: OpenAIResponsesModelSettings::default(),
        }
    }

    /// Create from environment variable `OPENAI_API_KEY`.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            ModelError::Configuration("OPENAI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(model_name, api_key))
    }

    /// Set the base URL.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set default model-specific settings.
    #[must_use]
    pub fn with_settings(mut self, settings: OpenAIResponsesModelSettings) -> Self {
        self.default_settings = settings;
        self
    }

    /// Set the reasoning effort level.
    ///
    /// Accepts a [`ReasoningEffort`] or any string; known efforts map to
    /// named variants and unknown strings pass through verbatim.
    #[must_use]
    pub fn with_reasoning_effort(mut self, effort: impl Into<ReasoningEffort>) -> Self {
        self.default_settings.reasoning_effort = Some(effort.into());
        self
    }

    /// Set the reasoning summary format.
    #[must_use]
    pub fn with_reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.default_settings.reasoning_summary = Some(summary);
        self
    }

    /// Set the organization ID.
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the project ID.
    #[must_use]
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set a custom HTTP client.
    #[must_use]
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    /// Set the default timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set a custom profile.
    #[must_use]
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Get the appropriate profile for a model name.
    fn profile_for_model(model: &str) -> ModelProfile {
        // All Responses API models support reasoning
        if model.starts_with("o1") || model.starts_with("o3") || model.contains("gpt-5") {
            openai_o1_profile()
        } else {
            // Default to o1 profile for responses API
            openai_o1_profile()
        }
    }

    /// Convert our messages to Responses API input format.
    fn map_messages(
        &self,
        messages: &[ModelRequest],
        _settings: &OpenAIResponsesModelSettings,
    ) -> (Vec<ResponseInput>, Option<String>) {
        let mut inputs = Vec::new();
        let mut instructions = None;

        for req in messages {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        // System prompts become instructions in Responses API
                        instructions = Some(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        inputs.push(self.convert_user_prompt(user));
                    }
                    ModelRequestPart::ToolReturn(tool_ret) => {
                        inputs.push(self.convert_tool_return(tool_ret));
                    }
                    ModelRequestPart::RetryPrompt(retry) => {
                        inputs.push(self.convert_retry_prompt(retry));
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        // Convert builtin tool return to tool input
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        inputs.push(ResponseInput::Tool {
                            tool_call_id: builtin.tool_call_id.clone(),
                            content: content_str,
                        });
                    }
                    ModelRequestPart::ModelResponse(response) => {
                        // Add the assistant response to inputs for proper alternation
                        inputs.extend(self.convert_response_to_inputs(response));
                    }
                }
            }
        }

        (inputs, instructions)
    }

    fn convert_user_prompt(&self, user: &UserPromptPart) -> ResponseInput {
        let content = self.convert_user_content(&user.content);
        ResponseInput::User { content }
    }

    fn convert_user_content(&self, content: &UserContent) -> ResponseInputContent {
        match content {
            UserContent::Text(text) => ResponseInputContent::Text(text.clone()),
            UserContent::Parts(parts) => {
                let converted: Vec<ResponseInputPart> = parts
                    .iter()
                    .filter_map(|p| self.convert_content_part(p))
                    .collect();
                if converted.len() == 1 {
                    if let ResponseInputPart::Text { text } = &converted[0] {
                        return ResponseInputContent::Text(text.clone());
                    }
                }
                ResponseInputContent::Parts(converted)
            }
        }
    }

    fn convert_content_part(&self, part: &UserContentPart) -> Option<ResponseInputPart> {
        match part {
            UserContentPart::Text { text } => Some(ResponseInputPart::Text { text: text.clone() }),
            UserContentPart::Image { image } => {
                let url = match image {
                    ImageContent::Url(u) => u.url.clone(),
                    ImageContent::Binary(b) => {
                        format!(
                            "data:{};base64,{}",
                            b.media_type.mime_type(),
                            base64::engine::general_purpose::STANDARD.encode(&b.data)
                        )
                    }
                };
                Some(ResponseInputPart::ImageUrl {
                    image_url: url,
                    detail: None,
                })
            }
            _ => None, // Skip unsupported types
        }
    }

    fn convert_tool_return(&self, tool_ret: &ToolReturnPart) -> ResponseInput {
        ResponseInput::Item(serde_json::json!({
            "type":"function_call_output", "call_id":tool_ret.tool_call_id,
            "output":tool_ret.content.to_string_content(),
        }))
    }

    fn convert_retry_prompt(&self, retry: &RetryPromptPart) -> ResponseInput {
        ResponseInput::User {
            content: ResponseInputContent::Text(retry.content.message().to_string()),
        }
    }

    /// Replay native items in their original order, not as invented Chat fields.
    fn convert_response_to_inputs(&self, response: &ModelResponse) -> Vec<ResponseInput> {
        if let Some(items) = super::responses_metadata::passive_replay(response) {
            return items.into_iter().map(ResponseInput::Item).collect();
        }
        response
            .parts
            .iter()
            .filter_map(|part| match part {
                ModelResponsePart::Text(text) => Some(ResponseInput::Assistant {
                    content: ResponseInputContent::Text(text.content.clone()),
                    reasoning_id: None,
                }),
                ModelResponsePart::Thinking(thinking)
                    if thinking.provider_name.as_deref() == Some("openai") =>
                {
                    let mut item = serde_json::json!({"type":"reasoning", "summary": []});
                    if let Some(id) = &thinking.id {
                        item["id"] = id.clone().into();
                    }
                    if let Some(details) = &thinking.provider_details {
                        if let Some(summary) = details.get("summary") {
                            item["summary"] = summary.clone();
                        }
                    }
                    if let Some(encrypted) = &thinking.signature {
                        item["encrypted_content"] = encrypted.clone().into();
                    }
                    Some(ResponseInput::Item(item))
                }
                ModelResponsePart::ToolCall(tool) => Some(ResponseInput::Item(serde_json::json!({
                    "type":"function_call", "call_id":tool.tool_call_id, "name":tool.tool_name,
                    "arguments": match &tool.args {
                        ToolCallArgs::String(value) => value.clone(),
                        ToolCallArgs::Json(value) => value.to_string(),
                    }
                }))),
                _ => None,
            })
            .collect()
    }

    /// Convert tool definitions to Responses API format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<ResponseTool> {
        tools
            .iter()
            .map(|t| {
                let params = serde_json::to_value(&t.parameters_json_schema)
                    .unwrap_or(serde_json::json!({}));

                ResponseTool::Function {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: params,
                    strict: t.strict,
                }
            })
            .collect()
    }

    /// Build the request body.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> ResponsesApiRequest {
        let (input, instructions) = self.map_messages(messages, &self.default_settings);

        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools(&params.tools))
        };

        let reasoning = if self.default_settings.reasoning_effort.is_some()
            || self.default_settings.reasoning_summary.is_some()
        {
            Some(ReasoningConfig {
                effort: self.default_settings.reasoning_effort.clone(),
                summary: self.default_settings.reasoning_summary,
            })
        } else {
            None
        };

        let truncation = self
            .default_settings
            .truncation
            .map(|t| TruncationConfig { truncation_type: t });

        ResponsesApiRequest {
            model: self.model_name.clone(),
            include: vec!["reasoning.encrypted_content".to_owned()],
            input,
            instructions,
            tools,
            reasoning,
            max_output_tokens: settings.max_tokens,
            temperature: settings.temperature,
            top_p: settings.top_p,
            stream,
            previous_response_id: self.default_settings.previous_response_id.clone(),
            service_tier: self.default_settings.service_tier,
            truncation,
            user: None,
            store: None,
            metadata: None,
        }
    }

    /// Parse the Responses API response into our format.
    fn process_response(&self, resp: ResponsesApiResponse) -> Result<ModelResponse, ModelError> {
        // Check for errors
        if resp.status == ResponseStatus::Failed {
            if let Some(err) = resp.error {
                return Err(ModelError::Api {
                    message: err.message,
                    code: Some(err.code),
                });
            }
            return Err(ModelError::api("Response failed with unknown error"));
        }

        match resp.status {
            ResponseStatus::Cancelled => return Err(ModelError::Cancelled),
            ResponseStatus::InProgress => {
                return Err(ModelError::incomplete_stream(
                    "Responses endpoint returned a non-terminal response",
                ));
            }
            _ => {}
        }
        let mut parts = Vec::new();
        let mut refused = false;

        for output in resp.output {
            match output {
                ResponseOutputItem::Reasoning {
                    id,
                    summary,
                    encrypted_content,
                    ..
                } => {
                    // Convert reasoning to ThinkingPart
                    let content: String = summary
                        .iter()
                        .map(|s| match s {
                            ReasoningSummaryItem::Text { text } => text.as_str(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    {
                        let mut thinking = ThinkingPart::new(content)
                            .with_id(&id)
                            .with_provider_name("openai");
                        thinking.signature = encrypted_content;
                        thinking.provider_details = Some(serde_json::Map::from_iter([(
                            "summary".to_owned(),
                            serde_json::to_value(&summary).unwrap_or_default(),
                        )]));
                        parts.push(ModelResponsePart::Thinking(thinking));
                    }
                }
                ResponseOutputItem::Message { content, .. } => {
                    for item in content {
                        match item {
                            MessageContentItem::Text { text, .. } => {
                                if !text.is_empty() {
                                    parts.push(ModelResponsePart::Text(TextPart::new(text)));
                                }
                            }
                            MessageContentItem::Refusal { .. } => {
                                refused = true;
                            }
                        }
                    }
                }
                ResponseOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    let args: JsonValue = serde_json::from_str(&arguments).map_err(|_| {
                        ModelError::invalid_response("Invalid Responses function-call arguments")
                    })?;
                    parts.push(ModelResponsePart::ToolCall(
                        ToolCallPart::new(name, ToolCallArgs::Json(args))
                            .with_tool_call_id(call_id),
                    ));
                }
                // Built-in tool results are typically internal, but we can expose them
                ResponseOutputItem::WebSearchCall { .. }
                | ResponseOutputItem::CodeInterpreterCall { .. }
                | ResponseOutputItem::FileSearchCall { .. }
                | ResponseOutputItem::ImageGenerationCall { .. }
                | ResponseOutputItem::McpCall { .. }
                | ResponseOutputItem::FunctionCallOutput { .. } => {
                    // These are intermediate results, usually not exposed to user
                }
            }
        }

        let finish_reason = if refused {
            Some(FinishReason::ContentFilter)
        } else {
            match resp.status {
                ResponseStatus::Completed => Some(FinishReason::Stop),
                ResponseStatus::Incomplete => Some(
                    match resp
                        .incomplete_details
                        .as_ref()
                        .and_then(|v| v["reason"].as_str())
                    {
                        Some("max_output_tokens") => FinishReason::Length,
                        Some("content_filter") => FinishReason::ContentFilter,
                        _ => {
                            return Err(ModelError::invalid_response(
                                "Unknown Responses incomplete reason",
                            ));
                        }
                    },
                ),
                ResponseStatus::Cancelled => Some(FinishReason::Stop),
                _ => None,
            }
        };

        let usage = resp.usage.map(|u| RequestUsage {
            request_tokens: Some(u.input_tokens),
            response_tokens: Some(u.output_tokens),
            total_tokens: Some(u.total_tokens),
            cache_creation_tokens: None,
            cache_read_tokens: u.input_tokens_details.and_then(|d| d.cached_tokens),
            details: u.output_tokens_details.map(|d| {
                let mut map = serde_json::Map::new();
                if let Some(reasoning) = d.reasoning_tokens {
                    map.insert("reasoning_tokens".to_string(), reasoning.into());
                }
                JsonValue::Object(map)
            }),
        });

        Ok(ModelResponse {
            parts,
            model_name: Some(resp.model),
            timestamp: chrono::Utc::now(),
            finish_reason,
            usage,
            vendor_id: Some(resp.id),
            vendor_details: resp.metadata,
            kind: "response".to_string(),
        })
    }

    /// Handle API error response.
    fn handle_error_response(&self, status: u16, body: &str) -> ModelError {
        // Try to parse as OpenAI error
        if let Ok(err) = serde_json::from_str::<super::types::OpenAIError>(body) {
            let code = err.error.code.clone();

            if status == 401 {
                return ModelError::auth(err.error.message);
            }
            if status == 429 {
                return ModelError::rate_limited(None);
            }
            if status == 404 {
                return ModelError::NotFound(err.error.message);
            }

            return ModelError::Api {
                message: err.error.message,
                code,
            };
        }

        ModelError::http(status, body)
    }
}

#[async_trait]
impl Model for OpenAIResponsesModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "openai"
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let body = self.build_request(messages, settings, params, false);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let mut request = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(timeout);

        if let Some(ref org) = self.organization {
            request = request.header("OpenAI-Organization", org);
        }
        if let Some(ref project) = self.project {
            request = request.header("OpenAI-Project", project);
        }

        let response = request.json(&body).send().await?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error_response(status, &body));
        }

        let raw: JsonValue = response
            .json()
            .await
            .map_err(|_| ModelError::invalid_response("Invalid Responses JSON"))?;
        let metadata = super::responses_metadata::terminal(&raw)?;
        // Only native parts enter replay conversion. All other output records are
        // retained in bounded archival metadata, including future provider types.
        let mut parsed = raw.clone();
        if let Some(items) = parsed["output"].as_array_mut() {
            items.retain(|item| {
                matches!(
                    item["type"].as_str(),
                    Some("message" | "reasoning" | "function_call")
                )
            });
        }
        let resp: ResponsesApiResponse = serde_json::from_value(parsed)
            .map_err(|_| ModelError::invalid_response("Unsupported Responses JSON shape"))?;
        let mut response = self.process_response(resp)?;
        metadata.apply(&mut response);
        Ok(response)
    }

    /// Stream native Responses SSE incrementally.
    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        let body = self.build_request(messages, settings, params, true);
        let mut request = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "text/event-stream")
            .timeout(settings.timeout.unwrap_or(self.default_timeout));
        if let Some(org) = &self.organization {
            request = request.header("OpenAI-Organization", org);
        }
        if let Some(project) = &self.project {
            request = request.header("OpenAI-Project", project);
        }
        let response = request.json(&body).send().await?;
        if !response.status().is_success() {
            return Err(self.handle_error_response(
                response.status().as_u16(),
                &response.text().await.unwrap_or_default(),
            ));
        }
        Ok(Box::pin(super::responses_stream::ResponsesStream::new(
            response.bytes_stream(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_core::messages::ModelResponseStreamEvent;

    /// Every effort variant serializes as its API string; custom values
    /// pass through verbatim.
    #[test]
    fn test_reasoning_effort_serialization() {
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Low).unwrap(),
            "\"low\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::High).unwrap(),
            "\"high\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Minimal).unwrap(),
            "\"minimal\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::XHigh).unwrap(),
            "\"xhigh\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Max).unwrap(),
            "\"max\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Custom("ultra".to_string())).unwrap(),
            "\"ultra\""
        );
    }

    /// Known efforts parse case-insensitively; unknown strings become
    /// `Custom` with the original casing preserved.
    #[test]
    fn test_reasoning_effort_parse_maps_known_and_keeps_custom_verbatim() {
        assert_eq!(ReasoningEffort::parse("xhigh"), ReasoningEffort::XHigh);
        assert_eq!(ReasoningEffort::parse("MAX"), ReasoningEffort::Max);
        assert_eq!(ReasoningEffort::from("Minimal"), ReasoningEffort::Minimal);
        assert_eq!(
            ReasoningEffort::from(String::from("xhigh")),
            ReasoningEffort::XHigh
        );
        assert_eq!(
            ReasoningEffort::parse("UltraDeep"),
            ReasoningEffort::Custom("UltraDeep".to_string())
        );
        assert_eq!(ReasoningEffort::XHigh.to_string(), "xhigh");
    }

    #[test]
    fn test_reasoning_summary_serialization() {
        assert_eq!(
            serde_json::to_string(&ReasoningSummary::Concise).unwrap(),
            "\"concise\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningSummary::Detailed).unwrap(),
            "\"detailed\""
        );
    }

    #[test]
    fn test_model_creation() {
        let model = OpenAIResponsesModel::new("o3-mini", "sk-test");
        assert_eq!(model.name(), "o3-mini");
        assert_eq!(model.system(), "openai");
    }

    #[test]
    fn test_model_builder() {
        let model = OpenAIResponsesModel::new("o3-mini", "sk-test")
            .with_reasoning_effort(ReasoningEffort::High)
            .with_reasoning_summary(ReasoningSummary::Detailed)
            .with_base_url("https://custom.api.com/v1")
            .with_timeout(Duration::from_secs(600));

        assert_eq!(model.base_url, "https://custom.api.com/v1");
        assert_eq!(model.default_timeout, Duration::from_secs(600));
        assert_eq!(
            model.default_settings.reasoning_effort,
            Some(ReasoningEffort::High)
        );
    }

    /// The effort builder accepts strings, mapping known values to variants.
    #[test]
    fn test_model_builder_accepts_effort_strings() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test")
            .with_reasoning_effort("xhigh")
            .with_reasoning_effort("max");
        assert_eq!(
            model.default_settings.reasoning_effort,
            Some(ReasoningEffort::Max)
        );
    }

    /// A set effort reaches the request body as the reasoning config.
    #[test]
    fn test_build_request_serializes_reasoning_effort() {
        let model =
            OpenAIResponsesModel::new("gpt-5.1-pro", "sk-test").with_reasoning_effort("max");
        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello");

        let request = model.build_request(
            &[req],
            &ModelSettings::new(),
            &ModelRequestParameters::new(),
            false,
        );

        let json = serde_json::to_string(&request).unwrap();
        assert!(
            json.contains(r#""reasoning":{"effort":"max"}"#),
            "expected max effort in request, got: {json}"
        );
    }

    /// The extended-config path applies the configured effort and options
    /// to the built Responses model, not just the model selection.
    #[test]
    fn test_extended_config_applies_effort_to_responses_model() {
        let config = crate::ExtendedModelConfig::new()
            .with_api_key("test-key")
            .with_reasoning_effort("xhigh")
            .with_base_url("https://custom.api.com/v1");

        let model = crate::openai_responses_model_from_config("gpt-5.1", &config).unwrap();

        assert_eq!(
            model.default_settings.reasoning_effort,
            Some(ReasoningEffort::XHigh)
        );
        assert_eq!(model.base_url, "https://custom.api.com/v1");
    }

    #[test]
    fn test_response_tool_serialization() {
        let tool = ResponseTool::Function {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            strict: Some(true),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"search\""));
    }

    #[test]
    fn test_web_search_tool_serialization() {
        let tool = ResponseTool::WebSearch {
            search_context_size: Some("medium".to_string()),
            user_location: None,
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("web_search_preview"));
    }

    #[test]
    fn test_response_input_serialization() {
        let input = ResponseInput::User {
            content: ResponseInputContent::Text("Hello".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
    }

    #[test]
    fn test_tool_input_serialization() {
        let input = ResponseInput::Tool {
            tool_call_id: "call_123".to_string(),
            content: "Result: 42".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"role\":\"tool\""));
        assert!(json.contains("\"tool_call_id\":\"call_123\""));
    }

    // Streaming fallback over wiremock (real HTTP request path).

    /// Build a completed non-streaming Responses response body, with usage
    /// attached only when the scenario provides it.
    fn completed_response_body(usage: Option<serde_json::Value>) -> serde_json::Value {
        let mut body = serde_json::json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 1234567890,
            "model": "o3-mini",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "Hello", "annotations": [] }]
            }],
            "error": null,
            "metadata": null,
            "service_tier": null
        });
        if let Some(usage) = usage {
            body["usage"] = usage;
        }
        body
    }

    /// Serve the given response body from a wiremock server and collect the
    /// events `request_stream` yields against it.
    async fn stream_events_for(body: serde_json::Value) -> Vec<ModelResponseStreamEvent> {
        use futures::StreamExt;

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/responses"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(format!(
                        "data: {}\n\n",
                        serde_json::json!({"type":"response.completed", "response":body})
                    )),
            )
            .mount(&server)
            .await;

        let model = OpenAIResponsesModel::new("o3-mini", "sk-test").with_base_url(server.uri());
        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello");

        let mut stream = model
            .request_stream(
                &[req],
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
            )
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(result) = stream.next().await {
            events.push(result.unwrap());
        }
        events
    }

    /// A streamed request over real HTTP replays the buffered completed
    /// response as part events and ends with exactly one terminal
    /// StreamComplete carrying the mapped finish reason and usage.
    #[tokio::test]
    async fn request_stream_emits_terminal_stream_complete_with_usage() {
        let usage = serde_json::json!({
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 3}
        });
        let events = stream_events_for(completed_response_body(Some(usage))).await;

        // Part events precede the terminal event.
        match events.first() {
            Some(ModelResponseStreamEvent::PartStart(start)) => {
                assert_eq!(start.index, 0);
                assert!(
                    matches!(&start.part, ModelResponsePart::Text(t) if t.content.is_empty()),
                    "expected the incremental text part start, got {:?}",
                    start.part
                );
            }
            other => panic!("expected a leading part event, got {:?}", other),
        }

        let terminals: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ModelResponseStreamEvent::StreamComplete(_)))
            .collect();
        assert_eq!(terminals.len(), 1, "expected exactly one terminal event");

        match events.last() {
            Some(ModelResponseStreamEvent::StreamComplete(complete)) => {
                assert_eq!(complete.finish_reason, FinishReason::Stop);
                assert_eq!(complete.input_tokens, Some(12));
                assert_eq!(complete.output_tokens, Some(7));
                assert_eq!(complete.cache_creation_tokens, None);
                assert_eq!(complete.cache_read_tokens, Some(4));
            }
            other => panic!("expected terminal StreamComplete last, got {:?}", other),
        }
    }

    /// A completed response without usage still ends with exactly one
    /// terminal event, and every token field stays `None`.
    #[tokio::test]
    async fn request_stream_without_usage_yields_none_token_fields() {
        let events = stream_events_for(completed_response_body(None)).await;

        let terminals: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ModelResponseStreamEvent::StreamComplete(_)))
            .collect();
        assert_eq!(terminals.len(), 1, "expected exactly one terminal event");

        match events.last() {
            Some(ModelResponseStreamEvent::StreamComplete(complete)) => {
                assert_eq!(complete.finish_reason, FinishReason::Stop);
                assert_eq!(complete.input_tokens, None);
                assert_eq!(complete.output_tokens, None);
                assert_eq!(complete.cache_creation_tokens, None);
                assert_eq!(complete.cache_read_tokens, None);
            }
            other => panic!("expected terminal StreamComplete last, got {:?}", other),
        }
    }
}
