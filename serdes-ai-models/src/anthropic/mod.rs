//! Anthropic Claude model implementations.
//!
//! This module provides complete implementations for Anthropic's Messages API:
//!
//! - [`AnthropicModel`]: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, etc.
//!
//! ## Features
//!
//! - **Extended Thinking**: Enable Claude's reasoning mode with `with_thinking()`
//! - **Prompt Caching**: Reduce costs with `with_caching()`
//! - **Multi-modal**: Images and documents (PDF) support
//! - **Tool Use**: Full function calling support
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_models::anthropic::AnthropicModel;
//! use serdes_ai_models::Model;
//!
//! let model = AnthropicModel::new(
//!     "claude-3-5-sonnet-20241022",
//!     std::env::var("ANTHROPIC_API_KEY").unwrap()
//! );
//!
//! // With extended thinking
//! let model = model.with_thinking(Some(10000));
//!
//! // With prompt caching
//! let model = model.with_caching();
//!
//! // Make a request
//! let response = model.request(&messages, &settings, &params).await?;
//! ```

pub mod model;
pub mod stream;
pub mod types;

// Re-exports
pub use model::AnthropicModel;
pub use types::{
    AnthropicContent, AnthropicError, AnthropicMessage, AnthropicTool, AnthropicToolChoice,
    AnthropicUsage, CacheControl, ContentBlock, ContentBlockDelta, ContentBlockStart,
    DocumentSource, ImageSource, MessagesRequest, MessagesResponse, ResponseContentBlock,
    StreamEvent, SystemBlock, SystemContent, ThinkingConfig, ToolResultBlock, ToolResultContent,
};

/// Create a new Anthropic Claude model.
///
/// This is a convenience function for creating an Anthropic model.
///
/// # Arguments
///
/// * `model_name` - The model name (e.g., "claude-3-5-sonnet-20241022")
/// * `api_key` - Your Anthropic API key
///
/// # Example
///
/// ```rust,ignore
/// use serdes_ai_models::anthropic;
///
/// let model = anthropic::claude("claude-3-5-sonnet-20241022", "sk-...");
/// ```
pub fn claude(model_name: impl Into<String>, api_key: impl Into<String>) -> AnthropicModel {
    AnthropicModel::new(model_name, api_key)
}

/// Common Anthropic model names.
pub mod models {
    /// Claude 3.5 Sonnet (latest, best for most tasks)
    pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-20241022";
    /// Claude 3.5 Haiku (fast, efficient)
    pub const CLAUDE_3_5_HAIKU: &str = "claude-3-5-haiku-20241022";
    /// Claude 3 Opus (most capable)
    pub const CLAUDE_3_OPUS: &str = "claude-3-opus-20240229";
    /// Claude 3 Sonnet
    pub const CLAUDE_3_SONNET: &str = "claude-3-sonnet-20240229";
    /// Claude 3 Haiku
    pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku-20240307";
}
