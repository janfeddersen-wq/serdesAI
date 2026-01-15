//! Antigravity (Google Cloud Code) model implementation.
//!
//! This module provides access to Gemini and Claude models via Google's
//! Antigravity API (Cloud Code). It uses OAuth authentication and supports
//! both streaming and non-streaming requests.
//!
//! # Overview
//!
//! Antigravity is Google's internal API that provides unified access to:
//! - Gemini models (gemini-2.5-pro, gemini-2.5-flash, gemini-3-flash, etc.)
//! - Claude models (claude-sonnet-4-5, etc.) via Anthropic partnership
//!
//! # Example
//!
//! ```rust,ignore
//! use serdes_ai_models::AntigravityModel;
//!
//! let model = AntigravityModel::new(
//!     "gemini-2.5-flash",
//!     "your-access-token",
//!     "your-project-id",
//! );
//!
//! // For Claude with thinking:
//! let claude = AntigravityModel::new(
//!     "claude-sonnet-4-5",
//!     "your-access-token",
//!     "your-project-id",
//! ).with_thinking(Some(10000));
//! ```

mod model;
mod stream;
mod types;

pub use model::AntigravityModel;
pub use types::{
    AntigravityConfig, AntigravityRequest, AntigravityResponse, Content, FunctionCall,
    FunctionDeclaration, FunctionResponse, GeminiRequest, GenerationConfig, Part,
    SystemInstruction, ThinkingConfig, Tool, ToolConfig, ANTIGRAVITY_ENDPOINT_AUTOPUSH,
    ANTIGRAVITY_ENDPOINT_DAILY, ANTIGRAVITY_ENDPOINT_PROD, DEFAULT_PROJECT_ID,
};
