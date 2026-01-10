//! OpenAI model implementations.
//!
//! This module provides complete implementations for OpenAI's API:
//!
//! - [`OpenAIChatModel`]: Chat completions (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
//! - [`OpenAIResponsesModel`]: Responses API for reasoning models (o1, o3, gpt-5)
//!
//! ## Example (Chat Completions)
//!
//! ```rust,ignore
//! use serdes_ai_models::openai::OpenAIChatModel;
//! use serdes_ai_models::Model;
//!
//! let model = OpenAIChatModel::new("gpt-4o", std::env::var("OPENAI_API_KEY").unwrap());
//!
//! // Make a request
//! let response = model.request(&messages, &settings, &params).await?;
//! ```
//!
//! ## Example (Responses API for Reasoning)
//!
//! ```rust,ignore
//! use serdes_ai_models::openai::{OpenAIResponsesModel, ReasoningEffort};
//! use serdes_ai_models::Model;
//!
//! let model = OpenAIResponsesModel::new("o3-mini", std::env::var("OPENAI_API_KEY").unwrap())
//!     .with_reasoning_effort(ReasoningEffort::High);
//!
//! // Make a request - reasoning content is returned as ThinkingPart
//! let response = model.request(&messages, &settings, &params).await?;
//! ```

pub mod chat;
pub mod responses;
pub mod stream;
pub mod types;

// Re-exports
pub use chat::OpenAIChatModel;
pub use responses::{
    OpenAIResponsesModel, OpenAIResponsesModelSettings, ReasoningEffort, ReasoningSummary,
    ServiceTier, TruncationMode,
};
pub use types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatTool,
    ContentPart, FunctionDefinition, MessageContent, ResponseFormat, ToolChoiceValue, Usage,
};

/// Create a new OpenAI chat model.
///
/// This is a convenience function for creating an OpenAI model.
///
/// # Arguments
///
/// * `model_name` - The model name (e.g., "gpt-4o", "gpt-4o-mini", "o1")
/// * `api_key` - Your OpenAI API key
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai_models::openai;
///
/// let model = openai::chat("gpt-4o", "sk-...");
/// ```
pub fn chat(model_name: impl Into<String>, api_key: impl Into<String>) -> OpenAIChatModel {
    OpenAIChatModel::new(model_name, api_key)
}

/// Create a new OpenAI Responses model (for o1, o3, gpt-5 reasoning models).
///
/// This is a convenience function for creating a Responses API model.
///
/// # Arguments
///
/// * `model_name` - The model name (e.g., "o1", "o3-mini", "gpt-5")
/// * `api_key` - Your OpenAI API key
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai_models::openai;
///
/// let model = openai::responses("o3-mini", "sk-...");
/// ```
pub fn responses(
    model_name: impl Into<String>,
    api_key: impl Into<String>,
) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new(model_name, api_key)
}

/// Common OpenAI model names.
pub mod models {
    /// GPT-4o (latest)
    pub const GPT_4O: &str = "gpt-4o";
    /// GPT-4o mini
    pub const GPT_4O_MINI: &str = "gpt-4o-mini";
    /// GPT-4 Turbo
    pub const GPT_4_TURBO: &str = "gpt-4-turbo";
    /// GPT-4
    pub const GPT_4: &str = "gpt-4";
    /// GPT-3.5 Turbo
    pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
    /// o1 (reasoning model) - use Responses API
    pub const O1: &str = "o1";
    /// o1-preview - use Responses API
    pub const O1_PREVIEW: &str = "o1-preview";
    /// o1-mini - use Responses API
    pub const O1_MINI: &str = "o1-mini";
    /// o3-mini (reasoning model) - use Responses API
    pub const O3_MINI: &str = "o3-mini";
    /// o3 (reasoning model) - use Responses API
    pub const O3: &str = "o3";
    /// gpt-5 (future) - use Responses API
    pub const GPT_5: &str = "gpt-5";
}
