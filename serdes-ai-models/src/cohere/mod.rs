//! Cohere model implementation.
//!
//! [Cohere](https://cohere.com) provides powerful language models including
//! the Command-R series with native RAG and tool-use capabilities.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_models::cohere::CohereModel;
//!
//! // Create from API key
//! let model = CohereModel::new("command-r-plus", api_key);
//!
//! // Or from environment (CO_API_KEY)
//! let model = CohereModel::from_env("command-r-plus")?;
//!
//! // Convenience constructors
//! let model = CohereModel::command_r_plus(api_key);
//! let model = CohereModel::command_r(api_key);
//! ```
//!
//! ## Available Models
//!
//! - `command-r-plus` - Most capable model, 128K context
//! - `command-r` - Balanced performance, 128K context
//! - `command` - Legacy model
//! - `command-light` - Faster, smaller model
//!
//! ## API Notes
//!
//! Cohere uses a unique API format:
//! - Base URL: `https://api.cohere.ai/v2`
//! - Auth: Bearer token via `CO_API_KEY` env var
//! - Uses `message` + `chat_history` instead of OpenAI-style messages array
//! - Supports tool calling and streaming

pub mod model;
pub mod types;

pub use model::CohereModel;
pub use types::{ChatMessage, ChatRequest, ChatResponse, Role, Tool, ToolCall};

/// Common Cohere model identifiers.
pub mod models {
    /// Command-R Plus - most capable model.
    pub const COMMAND_R_PLUS: &str = "command-r-plus";
    /// Command-R - balanced performance.
    pub const COMMAND_R: &str = "command-r";
    /// Command - legacy model.
    pub const COMMAND: &str = "command";
    /// Command Light - faster, smaller.
    pub const COMMAND_LIGHT: &str = "command-light";
}
