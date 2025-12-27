//! Agent-specific error types.
//!
//! This module defines all errors that can occur during agent execution.

use serdes_ai_models::ModelError;
use serdes_ai_tools::ToolError;
use thiserror::Error;

/// Errors that can occur during agent run execution.
#[derive(Debug, Error)]
pub enum AgentRunError {
    /// Model returned an error.
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Tool execution failed.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Output validation failed after retries.
    #[error("Output validation failed: {0}")]
    OutputValidationFailed(#[source] OutputValidationError),

    /// Failed to parse model output.
    #[error("Output parsing failed: {0}")]
    OutputParseFailed(#[source] OutputParseError),

    /// Usage limit was exceeded.
    #[error("Usage limit exceeded: {0}")]
    UsageLimitExceeded(#[from] UsageLimitError),

    /// Model stopped without producing output.
    #[error("Model stopped unexpectedly without output")]
    UnexpectedStop,

    /// No output was produced after all steps.
    #[error("No output produced")]
    NoOutput,

    /// Maximum retries exceeded.
    #[error("Max retries exceeded: {message}")]
    MaxRetriesExceeded {
        /// Description of what was being retried.
        message: String,
    },

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Agent was cancelled.
    #[error("Agent run was cancelled")]
    Cancelled,

    /// Timeout occurred.
    #[error("Agent run timed out after {seconds}s")]
    Timeout {
        /// Timeout in seconds.
        seconds: u64,
    },

    /// Provider error.
    #[error("Provider error: {0}")]
    Provider(String),

    /// Other error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AgentRunError {
    /// Create a max retries error.
    pub fn max_retries(message: impl Into<String>) -> Self {
        Self::MaxRetriesExceeded {
            message: message.into(),
        }
    }

    /// Create a configuration error.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }

    /// Create a timeout error.
    pub fn timeout(seconds: u64) -> Self {
        Self::Timeout { seconds }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Model(e) => e.is_retryable(),
            Self::Tool(e) => e.is_retryable(),
            Self::UsageLimitExceeded(_) => false,
            Self::Cancelled => false,
            Self::Timeout { .. } => false,
            Self::MaxRetriesExceeded { .. } => false,
            _ => true,
        }
    }
}

/// Output validation error.
#[derive(Debug, Error)]
pub enum OutputValidationError {
    /// Validation rule failed.
    #[error("Validation failed: {message}")]
    ValidationFailed {
        /// Error message.
        message: String,
        /// Field that failed (if applicable).
        field: Option<String>,
    },

    /// Custom validator failed.
    #[error("Custom validation failed: {0}")]
    Custom(String),

    /// Type mismatch.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type.
        expected: String,
        /// Actual type.
        actual: String,
    },

    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(String),
}

impl OutputValidationError {
    /// Create a validation failed error.
    pub fn failed(message: impl Into<String>) -> Self {
        Self::ValidationFailed {
            message: message.into(),
            field: None,
        }
    }

    /// Create a validation failed error for a specific field.
    pub fn field_failed(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ValidationFailed {
            message: message.into(),
            field: Some(field.into()),
        }
    }

    /// Create a custom validation error.
    pub fn custom(message: impl Into<String>) -> Self {
        Self::Custom(message.into())
    }

    /// Get the error message for retry prompts.
    pub fn retry_message(&self) -> String {
        match self {
            Self::ValidationFailed { message, field } => {
                if let Some(f) = field {
                    format!("Validation failed for field '{}': {}", f, message)
                } else {
                    format!("Validation failed: {}", message)
                }
            }
            Self::Custom(msg) => msg.clone(),
            Self::TypeMismatch { expected, actual } => {
                format!("Type mismatch: expected {}, got {}", expected, actual)
            }
            Self::MissingField(field) => {
                format!("Missing required field: {}", field)
            }
        }
    }
}

/// Output parsing error.
#[derive(Debug, Error)]
pub enum OutputParseError {
    /// JSON parsing failed.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// No output found in response.
    #[error("No output found in model response")]
    NotFound,

    /// Output tool not called.
    #[error("Output tool was not called by the model")]
    ToolNotCalled,

    /// Invalid format.
    #[error("Invalid output format: {0}")]
    InvalidFormat(String),

    /// Schema mismatch.
    #[error("Output does not match schema: {0}")]
    SchemaMismatch(String),
}

impl OutputParseError {
    /// Create an invalid format error.
    pub fn invalid_format(message: impl Into<String>) -> Self {
        Self::InvalidFormat(message.into())
    }

    /// Create a schema mismatch error.
    pub fn schema_mismatch(message: impl Into<String>) -> Self {
        Self::SchemaMismatch(message.into())
    }
}

/// Usage limit error.
#[derive(Debug, Error)]
pub enum UsageLimitError {
    /// Request token limit exceeded.
    #[error("Request token limit exceeded: {used} > {limit}")]
    RequestTokens {
        /// Tokens used.
        used: u64,
        /// Token limit.
        limit: u64,
    },

    /// Response token limit exceeded.
    #[error("Response token limit exceeded: {used} > {limit}")]
    ResponseTokens {
        /// Tokens used.
        used: u64,
        /// Token limit.
        limit: u64,
    },

    /// Total token limit exceeded.
    #[error("Total token limit exceeded: {used} > {limit}")]
    TotalTokens {
        /// Tokens used.
        used: u64,
        /// Token limit.
        limit: u64,
    },

    /// Request count limit exceeded.
    #[error("Request count limit exceeded: {count} > {limit}")]
    RequestCount {
        /// Number of requests.
        count: u32,
        /// Request limit.
        limit: u32,
    },

    /// Tool call limit exceeded.
    #[error("Tool call limit exceeded: {count} > {limit}")]
    ToolCalls {
        /// Number of tool calls.
        count: u32,
        /// Tool call limit.
        limit: u32,
    },

    /// Time limit exceeded.
    #[error("Time limit exceeded: {elapsed_seconds}s > {limit_seconds}s")]
    TimeLimit {
        /// Elapsed time in seconds.
        elapsed_seconds: u64,
        /// Time limit in seconds.
        limit_seconds: u64,
    },
}

/// Agent build error.
#[derive(Debug, Error)]
pub enum AgentBuildError {
    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(&'static str),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Tool registration error.
    #[error("Tool registration error: {0}")]
    ToolRegistration(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_run_error_display() {
        let err = AgentRunError::NoOutput;
        assert_eq!(err.to_string(), "No output produced");

        let err = AgentRunError::max_retries("tool call");
        assert!(err.to_string().contains("tool call"));
    }

    #[test]
    fn test_output_validation_error() {
        let err = OutputValidationError::failed("invalid value");
        assert!(err.retry_message().contains("invalid value"));

        let err = OutputValidationError::field_failed("name", "too short");
        assert!(err.retry_message().contains("name"));
        assert!(err.retry_message().contains("too short"));
    }

    #[test]
    fn test_usage_limit_error() {
        let err = UsageLimitError::TotalTokens {
            used: 1000,
            limit: 500,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));
    }

    #[test]
    fn test_is_retryable() {
        assert!(!AgentRunError::Cancelled.is_retryable());
        assert!(!AgentRunError::timeout(60).is_retryable());
        assert!(AgentRunError::NoOutput.is_retryable());
    }
}
