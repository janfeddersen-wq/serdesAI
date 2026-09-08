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

/// The Open Responses wire model: request/response types, tools, and error
/// envelopes defined by the Open Responses specification.
pub mod wire;

/// The Open Responses streaming event model and its translation onto
/// serdesAI model stream events.
pub mod events;

mod convert;

mod http;

mod session;

pub use session::Transport;

#[cfg(feature = "responses-ws")]
mod ws;

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::{ModelProfile, openai_o1_profile};
use async_trait::async_trait;
use convert::{history_to_wire, parts_from_output, tool_choice_to_wire, tool_to_wire};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serdes_ai_core::messages::{ModelResponseStreamEvent, PartStartEvent, StreamCompleteEvent};
use serdes_ai_core::{FinishReason, ModelRequest, ModelResponse, ModelSettings, RequestUsage};
use session::SharedConv;
use std::collections::HashMap;
use std::sync::Arc;
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
///
/// Shared by the HTTP path and the websocket `response.create` frame: both
/// transports serialize this one type, so their request shapes cannot
/// drift. `stream`, `store`, and `service_tier` are transport concerns and
/// stay `None` where a transport must not send them.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesApiRequest {
    /// Model to use.
    pub model: String,
    /// Conversation input items. Always a list on the wire, including when
    /// empty: an empty history serializes to `[]`, and the API rejects the
    /// empty string a text input would serialize to. (A history that adds
    /// nothing to a chained conversation is not an empty turn — the chain
    /// resets and the full input is re-sent; see `session::Conv::plan`.)
    pub input: Vec<wire::InputItem>,
    /// System instructions (replaces system message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<wire::ResponsesTool>>,
    /// Tool selection strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<wire::ResponsesToolChoice>,
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
    /// Whether to stream the response. HTTP always sends the key;
    /// websocket frames omit it entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Whether tool calls may run in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
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
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ResponseOutputItem {
    /// Reasoning/thinking output.
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<ReasoningSummaryItem>,
        /// Encrypted reasoning payload, replayed on chained turns so
        /// stateless requests keep the model's reasoning context.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
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
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ReasoningSummaryItem {
    #[serde(rename = "summary_text")]
    Text { text: String },
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

/// Transport-specific pieces of one request: what HTTP and websocket turns
/// legitimately differ on.
///
/// Everything else — input items, instructions, tools, tool choice,
/// reasoning, sampling — flows through the same shared mapping so the two
/// transports cannot drift apart.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestOverlay {
    /// Requests already delivered on the session and excluded from the
    /// input (continuation turns on both transports).
    pub(crate) skip: usize,
    /// The `stream` key: HTTP always sends it, websocket frames omit it.
    pub(crate) stream: Option<bool>,
    /// HTTP chaining sends true, stateless HTTP omits it, websocket sends false.
    pub(crate) store: Option<bool>,
    /// Continuation id: HTTP carries the configured default, websocket the
    /// session chain.
    pub(crate) previous_response_id: Option<String>,
    /// Service tier: HTTP honors the model settings, the websocket
    /// transport has never sent it.
    pub(crate) service_tier: Option<ServiceTier>,
    /// Truncation: HTTP honors the model settings, the websocket transport
    /// has never sent it.
    pub(crate) truncation: Option<TruncationConfig>,
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
    /// Transport used to reach the endpoint; HTTP is the default so the
    /// model's behavior is unchanged unless the websocket transport is
    /// selected explicitly.
    transport: Transport,
    /// Headers applied to both the websocket handshake and HTTP requests.
    headers: Vec<(String, String)>,
    /// Whether turns chain per conversation (session state, delta-only
    /// continuation input).
    chaining: bool,
    /// How long an untouched conversation is kept before it is evicted —
    /// and its websocket, if any, closed — at the next conversation
    /// lookup.
    conversation_idle_ttl: Duration,
    /// Conversation state, keyed by the initial-prefix fingerprints. The map lock
    /// guards lookup, insert, and the idle-eviction scan (no await is
    /// performed under it); each conversation serializes its own turns on
    /// its lock.
    conversations: Arc<std::sync::Mutex<HashMap<Vec<u64>, SharedConv>>>,
}

/// Map the model's reasoning settings onto the request's reasoning config.
///
/// Absent settings serialize nothing at all, matching an unconfigured
/// request.
fn reasoning_config(settings: &OpenAIResponsesModelSettings) -> Option<ReasoningConfig> {
    if settings.reasoning_effort.is_none() && settings.reasoning_summary.is_none() {
        return None;
    }
    Some(ReasoningConfig {
        effort: settings.reasoning_effort.clone(),
        summary: settings.reasoning_summary,
    })
}

impl OpenAIResponsesModel {
    /// Create a new OpenAI Responses model.
    ///
    /// The API key authenticates both HTTP requests and WebSocket handshakes.
    /// An explicit `Authorization` header supplied with [`Self::with_header`]
    /// overrides it on either transport.
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
            transport: Transport::Http,
            headers: Vec::new(),
            chaining: false,
            conversation_idle_ttl: session::CONVERSATION_IDLE_TTL,
            conversations: Arc::new(std::sync::Mutex::new(HashMap::new())),
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

    /// Set the organization ID for HTTP requests and WebSocket handshakes.
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the project ID for HTTP requests and WebSocket handshakes.
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

    /// Select the transport used to reach the endpoint.
    ///
    /// The websocket transport dials the base URL verbatim as the responses
    /// endpoint (a full `wss://…/v1/responses` URL) and requires the
    /// `responses-ws` feature; requests fail fast with a configuration
    /// error when the feature is compiled out. HTTP (the default) appends
    /// `/responses` to the base URL as before.
    #[must_use]
    pub fn with_transport(mut self, transport: Transport) -> Self {
        self.transport = transport;
        self
    }

    /// Set a header applied to both the websocket handshake and HTTP requests.
    ///
    /// Explicit headers override the API key's `Authorization` header and the
    /// organization/project headers, regardless of builder call order. Names
    /// are case-insensitive; the last `with_header` call for a name wins.
    /// WebSocket headers apply when a connection is opened, not on each turn
    /// of an already-open connection.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Resolve auth, routing, and explicit overrides before either transport
    /// applies headers, so HTTP appending and WebSocket insertion agree.
    fn request_headers(&self) -> Vec<(String, String)> {
        let mut headers = vec![(
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        )];
        if let Some(org) = &self.organization {
            headers.push(("OpenAI-Organization".to_string(), org.clone()));
        }
        if let Some(project) = &self.project {
            headers.push(("OpenAI-Project".to_string(), project.clone()));
        }
        for (name, value) in &self.headers {
            if let Some((_, existing)) = headers
                .iter_mut()
                .find(|(existing, _)| existing.eq_ignore_ascii_case(name))
            {
                existing.clone_from(value);
            } else {
                headers.push((name.clone(), value.clone()));
            }
        }
        headers
    }

    /// Enable or disable conversation-keyed session chaining.
    ///
    /// With chaining on, turns keep per-conversation session state
    /// (`previous_response_id` plus the requests already delivered) and
    /// send only each turn's new input items, on both transports: the
    /// websocket keeps a live socket per conversation with `store: false`,
    /// HTTP persists every turn with `store: true` and streams SSE.
    ///
    /// Lookup fingerprints the leading system-only requests plus the first
    /// request containing other material. Preserve that prefix across turns.
    /// Independent histories sharing that entire prefix cannot be identified
    /// by content; use separate model instances (not clones) for them.
    /// Replayed identical histories and changed sent prefixes reset the chain.
    /// System-only histories do not chain into histories with user material.
    /// This opt-in contract does not infer universal conversation identity.
    #[must_use]
    pub fn with_session_chaining(mut self, chaining: bool) -> Self {
        self.chaining = chaining;
        self
    }

    /// Set how long an untouched conversation is kept before lazy
    /// eviction.
    ///
    /// A conversation idle past this TTL is dropped — and its websocket,
    /// if any, closed with a proper handshake — at the next conversation
    /// lookup; the default is five minutes. Eviction runs only on lookup;
    /// no background task is spawned.
    #[must_use]
    pub fn with_conversation_idle_ttl(mut self, ttl: Duration) -> Self {
        self.conversation_idle_ttl = ttl;
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

    /// Build the request body for a non-chaining HTTP turn.
    ///
    /// The mapping itself is shared with the websocket transport (see
    /// [`Self::compose_request`]); the non-chaining overlay pins the
    /// transport fields: full replay, `stream` always sent, `store`
    /// omitted, routing fields from the model settings. Chained turns
    /// compose through the `http` module's `build_chained_request`
    /// instead.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> Result<ResponsesApiRequest, ModelError> {
        self.compose_request(
            messages,
            settings,
            params,
            RequestOverlay {
                skip: 0,
                stream: Some(stream),
                store: None,
                previous_response_id: self.default_settings.previous_response_id.clone(),
                service_tier: self.default_settings.service_tier,
                truncation: self
                    .default_settings
                    .truncation
                    .map(|t| TruncationConfig { truncation_type: t }),
            },
        )
    }

    /// Compose the request body from the shared mapping plus a transport
    /// overlay.
    ///
    /// History converts via [`history_to_wire`], tools via [`tool_to_wire`],
    /// and tool choice via [`tool_choice_to_wire`]; one malformed part
    /// (unsupported media, for instance) fails the request instead of being
    /// silently dropped.
    fn compose_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        overlay: RequestOverlay,
    ) -> Result<ResponsesApiRequest, ModelError> {
        let (instructions, input) = history_to_wire(messages, overlay.skip)?;

        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(params.tools.iter().map(tool_to_wire).collect())
        };

        Ok(ResponsesApiRequest {
            model: self.model_name.clone(),
            input,
            instructions,
            tools,
            tool_choice: tool_choice_to_wire(params.tool_choice.as_ref()),
            reasoning: reasoning_config(&self.default_settings),
            max_output_tokens: settings.max_tokens,
            temperature: settings.temperature,
            top_p: settings.top_p,
            stream: overlay.stream,
            parallel_tool_calls: settings.parallel_tool_calls,
            previous_response_id: overlay.previous_response_id,
            service_tier: overlay.service_tier,
            truncation: overlay.truncation,
            user: None,
            store: overlay.store,
            metadata: None,
        })
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

        let parts = parts_from_output(resp.output)?;

        let finish_reason = match resp.status {
            ResponseStatus::Completed => Some(FinishReason::Stop),
            ResponseStatus::Incomplete => Some(FinishReason::Length),
            ResponseStatus::Cancelled => Some(FinishReason::Stop),
            _ => None,
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

        // Open Responses error envelopes (a wire code without a `type`)
        // are provider errors like every other transport path produces;
        // only bodies without any envelope degrade to the bare status
        // error.
        if let Ok(envelope) = serde_json::from_str::<wire::HttpErrorEnvelope>(body) {
            return http::envelope_error(&envelope, status);
        }

        ModelError::http(status, body)
    }
}

/// Run one turn over the websocket transport, folding events into a
/// complete response.
#[cfg(feature = "responses-ws")]
async fn ws_request(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<ModelResponse, ModelError> {
    ws::request(model, messages, settings, params).await
}

/// The websocket transport is compiled out; fail fast with a clear error.
#[cfg(not(feature = "responses-ws"))]
async fn ws_request(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<ModelResponse, ModelError> {
    let _ = (model, messages, settings, params);
    Err(ModelError::Configuration(
        "websocket transport requires the responses-ws feature".to_string(),
    ))
}

/// Start a streamed turn over the websocket transport.
#[cfg(feature = "responses-ws")]
fn ws_stream(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<StreamedResponse, ModelError> {
    ws::stream(model, messages, settings, params)
}

/// The websocket transport is compiled out; fail fast with a clear error.
#[cfg(not(feature = "responses-ws"))]
fn ws_stream(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<StreamedResponse, ModelError> {
    let _ = (model, messages, settings, params);
    Err(ModelError::Configuration(
        "websocket transport requires the responses-ws feature".to_string(),
    ))
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
        match self.transport {
            Transport::WebSocket => ws_request(self, messages, settings, params).await,
            Transport::Http => {
                if self.chaining {
                    return http::request(self, messages, settings, params).await;
                }
                let body = self.build_request(messages, settings, params, false)?;

                let timeout = settings.timeout.unwrap_or(self.default_timeout);

                let response = http::post(self, &body, timeout).await?;

                let status = response.status().as_u16();
                if !response.status().is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(self.handle_error_response(status, &body));
                }

                let resp: ResponsesApiResponse = response
                    .json()
                    .await
                    .map_err(|e| ModelError::invalid_response(e.to_string()))?;

                self.process_response(resp)
            }
        }
    }

    /// Stream a response.
    ///
    /// The websocket transport streams wire events natively. Chained HTTP
    /// turns stream SSE natively as well; without chaining, HTTP keeps the
    /// non-streaming request fallback: the buffered completed response is
    /// replayed as part events and ends with one terminal event.
    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        match self.transport {
            Transport::WebSocket => ws_stream(self, messages, settings, params),
            Transport::Http => {
                if self.chaining {
                    return http::stream(self, messages, settings, params);
                }

                // Deliberate fallback: a stateless caller replays its full
                // input every turn, so buffering the completed response
                // costs nothing chaining would save; chained turns stream
                // SSE natively (see http::stream).
                let response = self.request(messages, settings, params).await?;

                let ModelResponse {
                    parts,
                    finish_reason,
                    usage,
                    ..
                } = response;

                // Part events first, terminal event last; the request error above
                // short-circuits failures before any event is emitted.
                let mut events: Vec<Result<ModelResponseStreamEvent, ModelError>> = parts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, part)| {
                        Ok(ModelResponseStreamEvent::PartStart(PartStartEvent::new(
                            idx, part,
                        )))
                    })
                    .collect();

                events.push(Ok(stream_complete_event(finish_reason, usage.as_ref())));

                Ok(Box::pin(futures::stream::iter(events)))
            }
        }
    }
}

/// Build the terminal event from the finish reason and usage the
/// non-streaming path mapped from the completed response.
///
/// Usage fields absent from the response stay `None`; a status the
/// non-streaming mapping leaves unmapped defaults to [`FinishReason::Stop`],
/// matching the chat stream parser's terminal default.
fn stream_complete_event(
    finish_reason: Option<FinishReason>,
    usage: Option<&RequestUsage>,
) -> ModelResponseStreamEvent {
    let mut event = StreamCompleteEvent::new(finish_reason.unwrap_or(FinishReason::Stop));

    if let Some(u) = usage {
        if let Some(tokens) = u.request_tokens {
            event = event.with_input_tokens(tokens);
        }
        if let Some(tokens) = u.response_tokens {
            event = event.with_output_tokens(tokens);
        }
        if let Some(tokens) = u.cache_creation_tokens {
            event = event.with_cache_creation_tokens(tokens);
        }
        if let Some(tokens) = u.cache_read_tokens {
            event = event.with_cache_read_tokens(tokens);
        }
    }

    ModelResponseStreamEvent::StreamComplete(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProviderErrorKind;
    use serdes_ai_core::ModelRequestPart;
    use serdes_ai_core::messages::{
        ModelResponsePart, RetryPromptPart, SystemPromptPart, TextPart, ThinkingPart, ToolCallArgs,
        ToolCallPart, ToolReturnPart, UserContent, UserContentPart, UserPromptPart, VideoContent,
    };

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

        let request = model
            .build_request(
                &[req],
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
                false,
            )
            .expect("request builds");

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

    /// A function tool serializes in the wire form with its `type` tag and
    /// strict flag.
    #[test]
    fn test_response_tool_serialization() {
        let tool = wire::ResponsesTool::Function {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            strict: Some(true),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"search\""));
    }

    /// Hosted tool types pass through as builtin tools carrying their wire
    /// tag; the receiving side decides whether it can execute them.
    #[test]
    fn test_web_search_tool_serialization() {
        let tool = wire::ResponsesTool::Builtin {
            tool_type: "web_search_preview".to_string(),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"web_search_preview\""));
    }

    /// User turns serialize as "easy input messages": role and content with
    /// no `type` tag, matching the API's easy-message shape.
    #[test]
    fn test_response_input_serialization() {
        let input = wire::InputItem::Easy(wire::EasyInputMessage {
            role: wire::InputRole::User,
            content: Some(wire::InputMessageContent::Text("Hello".to_string())),
        });
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
        assert!(
            !json.contains("input_text"),
            "plain text stays a string, got: {json}"
        );
    }

    /// Tool returns serialize as function_call_output items keyed by call
    /// id, not role:"tool" messages.
    #[test]
    fn test_tool_input_serialization() {
        let input = wire::InputItem::Typed(wire::TypedInputItem::FunctionCallOutput {
            call_id: "call_123".to_string(),
            output: "Result: 42".to_string(),
        });
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"type\":\"function_call_output\""));
        assert!(json.contains("\"call_id\":\"call_123\""));
        assert!(json.contains("\"output\":\"Result: 42\""));
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
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
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
                    matches!(&start.part, ModelResponsePart::Text(t) if t.content == "Hello"),
                    "expected the buffered text part first, got {:?}",
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

    // ------------------------------------------------------------------
    // S6a: wire-accurate request/response mapping pins
    // ------------------------------------------------------------------

    /// Build one model request carrying a single part.
    fn single_request(part: ModelRequestPart) -> ModelRequest {
        ModelRequest::with_parts(vec![part])
    }

    /// An assistant response carrying one of each replayable part kind.
    fn assistant_response() -> ModelResponse {
        let thinking = ThinkingPart::new("pondering")
            .with_provider_name("openai")
            .with_provider_details(
                [(
                    "encrypted_content".to_string(),
                    JsonValue::String("enc-1".to_string()),
                )]
                .into_iter()
                .collect(),
            );
        ModelResponse {
            parts: vec![
                ModelResponsePart::Thinking(thinking),
                ModelResponsePart::Text(TextPart::new("the answer")),
                ModelResponsePart::ToolCall(
                    ToolCallPart::new(
                        "get_weather",
                        ToolCallArgs::Json(serde_json::json!({ "city": "NYC" })),
                    )
                    .with_tool_call_id("call_9"),
                ),
            ],
            model_name: None,
            timestamp: chrono::Utc::now(),
            finish_reason: None,
            usage: None,
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        }
    }

    /// System prompts join into one instructions string. The legacy mapping
    /// let the last system prompt win; the shared mapping joins every
    /// system prompt, so none is silently lost.
    #[test]
    fn build_request_joins_system_prompts_into_instructions() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test");
        let history = [
            single_request(ModelRequestPart::SystemPrompt(SystemPromptPart::new(
                "be brief",
            ))),
            single_request(ModelRequestPart::SystemPrompt(SystemPromptPart::new(
                "answer in French",
            ))),
            single_request(ModelRequestPart::UserPrompt(UserPromptPart::new("Hello"))),
        ];

        let request = model
            .build_request(
                &history,
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
                false,
            )
            .expect("request builds");

        assert_eq!(
            request.instructions.as_deref(),
            Some("be brief\n\nanswer in French")
        );
        assert_eq!(request.input.len(), 1, "only the user turn is input");
    }

    /// Assistant history echoes back as typed wire items — a reasoning item
    /// with its encrypted content, an assistant message, and a function
    /// call — and retry prompts are dropped instead of becoming user text.
    #[test]
    fn build_request_echoes_assistant_history_as_typed_items() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test");
        let history = [
            single_request(ModelRequestPart::UserPrompt(UserPromptPart::new("hi"))),
            single_request(ModelRequestPart::ModelResponse(Box::new(
                assistant_response(),
            ))),
            single_request(ModelRequestPart::RetryPrompt(RetryPromptPart::new(
                "recover",
            ))),
            single_request(ModelRequestPart::UserPrompt(UserPromptPart::new("next"))),
        ];

        let request = model
            .build_request(
                &history,
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
                false,
            )
            .expect("request builds");

        // user + (reasoning, message, function call) + user; the retry
        // prompt would add a sixth item if it leaked into the input.
        assert_eq!(
            request.input.len(),
            5,
            "retry prompt dropped, all response parts replayed"
        );
        match &request.input[1] {
            wire::InputItem::Typed(wire::TypedInputItem::Reasoning {
                summary,
                encrypted_content,
                ..
            }) => {
                assert_eq!(summary.len(), 1);
                assert_eq!(summary[0].text, "pondering");
                assert_eq!(encrypted_content.as_deref(), Some("enc-1"));
            }
            other => panic!("expected a reasoning item, got {other:?}"),
        }
        match &request.input[3] {
            wire::InputItem::Typed(wire::TypedInputItem::FunctionCall {
                call_id, name, ..
            }) => {
                assert_eq!(call_id, "call_9");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected a function call item, got {other:?}"),
        }
    }

    /// Tool returns serialize on the wire as function_call_output items
    /// keyed by call id, not role:"tool" messages.
    #[test]
    fn build_request_maps_tool_returns_to_function_call_outputs() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test");
        let history = [single_request(ModelRequestPart::ToolReturn(
            ToolReturnPart::success("get_weather", "sunny").with_tool_call_id("call_9"),
        ))];

        let request = model
            .build_request(
                &history,
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
                false,
            )
            .expect("request builds");

        assert_eq!(request.input.len(), 1);
        match &request.input[0] {
            wire::InputItem::Typed(wire::TypedInputItem::FunctionCallOutput {
                call_id,
                output,
            }) => {
                assert_eq!(call_id, "call_9");
                assert_eq!(output, "sunny");
            }
            other => panic!("expected a function_call_output item, got {other:?}"),
        }
    }

    /// Unsupported media parts fail the request with an invalid-request
    /// error instead of being silently skipped.
    #[test]
    fn build_request_rejects_unsupported_media_parts() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test");
        let history = [single_request(ModelRequestPart::UserPrompt(
            UserPromptPart::new(UserContent::Parts(vec![UserContentPart::Video {
                video: VideoContent::url("https://example.com/clip.mp4"),
            }])),
        ))];

        let error = model
            .build_request(
                &history,
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
                false,
            )
            .expect_err("video input must be rejected");

        match error {
            ModelError::Provider { kind, .. } => {
                assert_eq!(kind, ProviderErrorKind::InvalidRequest)
            }
            other => panic!("expected an invalid-request provider error, got {other:?}"),
        }
    }

    /// The stream key is a transport concern: HTTP turns still send it
    /// (false for the buffered non-streaming request), websocket frames
    /// omit it entirely.
    #[test]
    fn stream_serializes_only_when_set() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test");
        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello");

        let http = model
            .build_request(
                &[req],
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
                false,
            )
            .expect("request builds");
        let http_json = serde_json::to_string(&http).unwrap();
        assert!(
            http_json.contains("\"stream\":false"),
            "HTTP requests keep the legacy stream key, got: {http_json}"
        );

        let ws = ResponsesApiRequest {
            model: "gpt-5.1".to_string(),
            input: Vec::new(),
            instructions: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            service_tier: None,
            truncation: None,
            user: None,
            store: None,
            metadata: None,
        };
        let ws_json = serde_json::to_string(&ws).unwrap();
        assert!(
            !ws_json.contains("stream"),
            "websocket frames omit stream, got: {ws_json}"
        );
    }

    /// Reasoning output stashes the encrypted content in provider details,
    /// and replaying the response as history puts it back on the wire
    /// reasoning item — the stateless chaining loop for reasoning models.
    #[test]
    fn reasoning_encrypted_content_survives_the_round_trip() {
        let model = OpenAIResponsesModel::new("gpt-5.1", "sk-test");
        let resp: ResponsesApiResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-5.1",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "hmm"}],
                    "encrypted_content": "enc-1",
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}]
                }
            ],
            "error": null
        }))
        .expect("response parses");

        let response = model.process_response(resp).expect("response maps");
        let thinking = response
            .parts
            .iter()
            .find_map(|part| match part {
                ModelResponsePart::Thinking(thinking) => Some(thinking),
                _ => None,
            })
            .expect("thinking part present");
        assert_eq!(
            thinking
                .provider_details
                .as_ref()
                .and_then(|details| details.get("encrypted_content"))
                .and_then(|value| value.as_str()),
            Some("enc-1")
        );

        let (_, items) = history_to_wire(
            &[single_request(ModelRequestPart::ModelResponse(Box::new(
                response,
            )))],
            0,
        )
        .expect("history converts");
        match &items[0] {
            wire::InputItem::Typed(wire::TypedInputItem::Reasoning {
                encrypted_content, ..
            }) => assert_eq!(encrypted_content.as_deref(), Some("enc-1")),
            other => panic!("expected a reasoning item, got {other:?}"),
        }
    }

    /// Over real HTTP the request body carries the wire item forms: input
    /// as an array whose user turn is an easy message, the legacy
    /// `stream:false` key, and no optional keys the caller did not set.
    #[tokio::test]
    async fn http_request_body_uses_wire_item_forms() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/responses"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(completed_response_body(None)),
            )
            .mount(&server)
            .await;

        let model = OpenAIResponsesModel::new("o3-mini", "sk-test").with_base_url(server.uri());
        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello");
        model
            .request(
                &[req],
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
            )
            .await
            .expect("turn ok");

        let received = server.received_requests().await.expect("requests recorded");
        assert_eq!(received.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();

        assert_eq!(body["stream"], false);
        let input = body["input"].as_array().expect("input is a list");
        assert_eq!(input.len(), 1);
        assert_eq!(
            input[0],
            serde_json::json!({"role": "user", "content": "Hello"})
        );
        assert!(
            input[0].get("type").is_none(),
            "user input stays an easy message, got: {}",
            input[0]
        );
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    /// An Open Responses error envelope on the non-chaining HTTP path
    /// surfaces as a provider error with the wire code; the bare transport
    /// error is reserved for bodies without an envelope.
    #[tokio::test]
    async fn http_error_envelope_maps_to_provider_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/responses"))
            .respond_with(
                wiremock::ResponseTemplate::new(404).set_body_json(serde_json::json!({
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "resp_x is gone"
                    }
                })),
            )
            .mount(&server)
            .await;

        let model = OpenAIResponsesModel::new("o3-mini", "sk-test").with_base_url(server.uri());
        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello");

        let error = model
            .request(
                &[req],
                &ModelSettings::new(),
                &ModelRequestParameters::new(),
            )
            .await
            .expect_err("turn must fail");

        match &error {
            ModelError::Provider { code, status, .. } => {
                assert_eq!(code, "previous_response_not_found");
                assert_eq!(*status, Some(404));
            }
            other => panic!("expected a provider error, got {other:?}"),
        }
    }
}
