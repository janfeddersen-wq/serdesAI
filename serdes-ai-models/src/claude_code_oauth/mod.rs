//! Claude Code OAuth model implementation.
//!
//! This model uses OAuth access tokens to authenticate with the Anthropic API
//! via the Claude Code OAuth flow.
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_models::claude_code_oauth::ClaudeCodeOAuthModel;
//!
//! // Token comes from your app's storage (this crate doesn't store tokens)
//! let access_token = load_token_from_storage()?;
//! let model = ClaudeCodeOAuthModel::new("claude-sonnet-4-20250514", access_token);
//!
//! let response = model.request(&messages, &settings, &params).await?;
//! ```

mod model;
pub mod stream;
mod types;

pub use model::ClaudeCodeOAuthModel;
pub use stream::ClaudeCodeStreamParser;
pub use types::*;

/// Default Claude Code models (dynamically discovered, these are common ones).
pub mod models {
    /// Claude Sonnet 4
    pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4-20250514";
    /// Claude 3.5 Sonnet
    pub const CLAUDE_35_SONNET: &str = "claude-3-5-sonnet-20241022";
    /// Claude 3.5 Haiku
    pub const CLAUDE_35_HAIKU: &str = "claude-3-5-haiku-20241022";
}
