//! Agent error types.

use thiserror::Error;

/// Errors that can occur during agent operations.
#[derive(Error, Debug)]
pub enum AgentError {
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Model error.
    #[error("Model error: {0}")]
    Model(#[from] serdes_ai_core::Error),

    /// Tool error.
    #[error("Tool error: {0}")]
    Tool(String),

    /// Maximum turns exceeded.
    #[error("Maximum turns ({0}) exceeded")]
    MaxTurnsExceeded(usize),

    /// Validation error.
    #[error("Validation error: {0}")]
    Validation(String),
}
