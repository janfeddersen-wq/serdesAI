//! Antigravity (Google Cloud Code) API types.
//!
//! Antigravity uses a wrapped Gemini-style format with project context.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ============================================================================
// Constants
// ============================================================================

/// Default Antigravity endpoint (production).
pub const ANTIGRAVITY_ENDPOINT_PROD: &str = "https://cloudcode-pa.googleapis.com";

/// Daily sandbox endpoint.
pub const ANTIGRAVITY_ENDPOINT_DAILY: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";

/// Autopush sandbox endpoint.
pub const ANTIGRAVITY_ENDPOINT_AUTOPUSH: &str = "https://autopush-cloudcode-pa.sandbox.googleapis.com";

/// Default fallback project ID.
pub const DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";

// ============================================================================
// Request Types
// ============================================================================

/// Wrapped request body for Antigravity API.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AntigravityRequest {
    /// Project ID.
    pub project: String,
    /// Model name (e.g., "gemini-3-flash", "claude-sonnet-4-5").
    pub model: String,
    /// The actual Gemini-style request.
    pub request: GeminiRequest,
    /// Request type (always "agent").
    pub request_type: String,
    /// User agent identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// Request ID for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl AntigravityRequest {
    /// Create a new Antigravity request.
    pub fn new(project: String, model: String, request: GeminiRequest) -> Self {
        Self {
            project,
            model,
            request,
            request_type: "agent".to_string(),
            user_agent: Some("antigravity".to_string()),
            request_id: Some(format!("agent-{}", uuid::Uuid::new_v4())),
        }
    }
}

/// Inner Gemini-style request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    /// Content messages.
    pub contents: Vec<Content>,
    /// System instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Generation configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Session ID for multi-turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl GeminiRequest {
    /// Create a new request with contents.
    pub fn new(contents: Vec<Content>) -> Self {
        Self {
            contents,
            system_instruction: None,
            tools: None,
            tool_config: None,
            generation_config: None,
            session_id: None,
        }
    }
}

/// System instruction with role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInstruction {
    /// Role (should be "user" for Antigravity).
    pub role: String,
    /// Parts containing the instruction text.
    pub parts: Vec<Part>,
}

impl SystemInstruction {
    /// Create a new system instruction.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            parts: vec![Part::Text { text: text.into() }],
        }
    }
}

/// Content (message) in conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// Role: "user" or "model".
    pub role: String,
    /// Content parts.
    pub parts: Vec<Part>,
}

impl Content {
    /// Create user content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            parts: vec![Part::Text { text: text.into() }],
        }
    }

    /// Create model content.
    pub fn model(text: impl Into<String>) -> Self {
        Self {
            role: "model".to_string(),
            parts: vec![Part::Text { text: text.into() }],
        }
    }

    /// Create content with multiple parts.
    pub fn user_parts(parts: Vec<Part>) -> Self {
        Self {
            role: "user".to_string(),
            parts,
        }
    }

    /// Create model content with multiple parts.
    pub fn model_parts(parts: Vec<Part>) -> Self {
        Self {
            role: "model".to_string(),
            parts,
        }
    }
}

/// Content part.
/// 
/// IMPORTANT: Order matters for untagged enums! More specific variants must come first.
/// - Thinking (requires both `thought` and `text`) must come before Text (only requires `text`)
/// - FunctionCall (requires `functionCall`) must come before ThoughtSignature
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    /// Thinking block (for Claude/Gemini models).
    /// Format: {"thought": true, "text": "thinking content"}
    /// Must come BEFORE Text since both have `text` field!
    Thinking {
        /// Boolean flag indicating this is a thinking block.
        thought: bool,
        /// The actual thinking content.
        text: String,
    },
    /// Function call from model (with optional thought signature).
    /// Must come before ThoughtSignature since FunctionCall can also have thoughtSignature.
    FunctionCall {
        /// The function call.
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
        /// Thought signature for multi-turn tool calls.
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    /// Thought signature only (for multi-turn conversations).
    ThoughtSignature {
        /// The thought signature.
        #[serde(rename = "thoughtSignature")]
        thought_signature: String,
    },
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Inline data (images, etc.).
    InlineData {
        /// The inline data.
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    /// Function response to model.
    FunctionResponse {
        /// The function response.
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
}

impl Part {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text { text: text.into() }
    }

    /// Create an inline data part.
    pub fn inline_data(mime_type: impl Into<String>, data: impl Into<String>) -> Self {
        Part::InlineData {
            inline_data: InlineData {
                mime_type: mime_type.into(),
                data: data.into(),
            },
        }
    }
}

/// Inline binary data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    /// MIME type.
    pub mime_type: String,
    /// Base64-encoded data.
    pub data: String,
}

/// Function call from model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Function arguments as JSON.
    pub args: JsonValue,
    /// Call ID for matching response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Function response to model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// Function name.
    pub name: String,
    /// Response content.
    pub response: JsonValue,
    /// Call ID to match with function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Function declarations.
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// Function declaration for tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// Parameter schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonValue>,
}

/// Tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Function calling config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

/// Function calling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallingConfig {
    /// Mode: AUTO, ANY, NONE.
    pub mode: String,
    /// Allowed function names (for ANY mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top P.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top K.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Max output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Thinking configuration (for Claude models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

/// Thinking configuration for Claude and Gemini models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Include thoughts in response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    /// Thinking budget tokens (for Claude models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u64>,
    /// Thinking level for Gemini 3 models (low, medium, high).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
}

// ============================================================================
// Response Types
// ============================================================================

/// Wrapped streaming response from Antigravity API.
/// The actual Gemini response is nested inside "response" field.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AntigravityResponse {
    /// The actual Gemini-style response.
    pub response: GeminiStreamResponse,
    /// Trace ID for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

/// Inner Gemini-style response from stream.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiStreamResponse {
    /// Candidates (response options).
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Usage metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    /// Model version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Response ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

/// Response candidate.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// Content of the response.
    #[serde(default)]
    pub content: Option<Content>,
    /// Finish reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Safety ratings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    /// Index in candidates array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
}

/// Safety rating.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// Category.
    pub category: String,
    /// Probability.
    pub probability: String,
}

/// Usage metadata.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Prompt token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<i32>,
    /// Candidates token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<i32>,
    /// Total token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<i32>,
    /// Cached content token count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<i32>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Antigravity model configuration.
#[derive(Debug, Clone)]
pub struct AntigravityConfig {
    /// Base API endpoint.
    pub endpoint: String,
    /// Fallback endpoints to try.
    pub fallback_endpoints: Vec<String>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for AntigravityConfig {
    fn default() -> Self {
        Self {
            // Daily sandbox first (same as CLIProxy/Vibeproxy)
            endpoint: ANTIGRAVITY_ENDPOINT_DAILY.to_string(),
            fallback_endpoints: vec![
                ANTIGRAVITY_ENDPOINT_AUTOPUSH.to_string(),
                ANTIGRAVITY_ENDPOINT_PROD.to_string(),
            ],
            timeout_secs: 180,
        }
    }
}
