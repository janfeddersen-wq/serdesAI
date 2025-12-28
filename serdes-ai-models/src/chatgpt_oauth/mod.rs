//! ChatGPT OAuth model implementation.
//!
//! This model uses OAuth access tokens to authenticate with the ChatGPT Codex API.
//! The API is OpenAI-compatible but uses a different endpoint (chatgpt.com/backend-api/codex).
//!
//! ## Example
//!
//! ```rust,ignore
//! use serdes_ai_models::chatgpt_oauth::ChatGptOAuthModel;
//!
//! // Token comes from your app's storage (this crate doesn't store tokens)
//! let access_token = load_token_from_storage()?;
//! let model = ChatGptOAuthModel::new("chatgpt-4o-codex", access_token);
//!
//! let response = model.request(&messages, &settings, &params).await?;
//! ```

mod model;
mod types;

pub use model::ChatGptOAuthModel;
pub use types::*;

/// Available ChatGPT Codex models.
pub mod models {
    /// ChatGPT 4o Codex
    pub const CHATGPT_4O_CODEX: &str = "chatgpt-4o-codex";
    /// ChatGPT o1 Codex  
    pub const CHATGPT_O1_CODEX: &str = "chatgpt-o1-codex";
    /// ChatGPT o3 Codex
    pub const CHATGPT_O3_CODEX: &str = "chatgpt-o3-codex";
    /// ChatGPT o4-mini Codex
    pub const CHATGPT_O4_MINI_CODEX: &str = "chatgpt-o4-mini-codex";
}
