//! Tool-specific error types.
//!
//! This module provides comprehensive error types for tool execution,
//! including retryable errors, approval flows, and validation failures.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during tool execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool execution failed.
    #[error("Tool execution failed: {message}")]
    ExecutionFailed {
        /// Error message.
        message: String,
        /// Whether this error is retryable.
        retryable: bool,
    },

    /// Invalid arguments provided to the tool.
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    /// Tool not found in registry.
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Model should retry with different arguments.
    #[error("Model retry requested: {0}")]
    ModelRetry(String),

    /// Tool requires approval before execution.
    #[error("Approval required for tool '{tool_name}'")]
    ApprovalRequired {
        /// Name of the tool requiring approval.
        tool_name: String,
        /// Arguments that were provided.
        args: serde_json::Value,
    },

    /// Tool call was deferred for later execution.
    #[error("Tool call '{tool_name}' deferred")]
    CallDeferred {
        /// Name of the deferred tool.
        tool_name: String,
        /// Arguments for the deferred call.
        args: serde_json::Value,
    },

    /// Tool execution timed out.
    #[error("Tool execution timed out after {0:?}")]
    Timeout(Duration),

    /// Argument validation failed.
    #[error("Argument validation failed: {message}")]
    ValidationFailed {
        /// Validation error message.
        message: String,
        /// Field that failed validation, if applicable.
        field: Option<String>,
    },

    /// Tool was cancelled.
    #[error("Tool execution cancelled")]
    Cancelled,

    /// Tool returned an error result.
    #[error("Tool returned error: {0}")]
    ToolReturnedError(String),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Other errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl ToolError {
    /// Check if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::ExecutionFailed { retryable, .. } => *retryable,
            Self::ModelRetry(_) => true,
            Self::Timeout(_) => true,
            Self::ValidationFailed { .. } => false,
            Self::InvalidArguments(_) => false,
            Self::NotFound(_) => false,
            Self::ApprovalRequired { .. } => false,
            Self::CallDeferred { .. } => false,
            Self::Cancelled => false,
            Self::ToolReturnedError(_) => false,
            Self::Json(_) => false,
            Self::Other(_) => false,
        }
    }

    /// Create a non-retryable execution failure.
    #[must_use]
    pub fn execution_failed(msg: impl Into<String>) -> Self {
        Self::ExecutionFailed {
            message: msg.into(),
            retryable: false,
        }
    }

    /// Create a retryable execution failure.
    #[must_use]
    pub fn retryable(msg: impl Into<String>) -> Self {
        Self::ExecutionFailed {
            message: msg.into(),
            retryable: true,
        }
    }

    /// Create an invalid arguments error.
    #[must_use]
    pub fn invalid_args(msg: impl Into<String>) -> Self {
        Self::InvalidArguments(msg.into())
    }

    /// Create a not found error.
    #[must_use]
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound(name.into())
    }

    /// Create a model retry error.
    #[must_use]
    pub fn model_retry(msg: impl Into<String>) -> Self {
        Self::ModelRetry(msg.into())
    }

    /// Create an approval required error.
    #[must_use]
    pub fn approval_required(tool_name: impl Into<String>, args: serde_json::Value) -> Self {
        Self::ApprovalRequired {
            tool_name: tool_name.into(),
            args,
        }
    }

    /// Create a call deferred error.
    #[must_use]
    pub fn call_deferred(tool_name: impl Into<String>, args: serde_json::Value) -> Self {
        Self::CallDeferred {
            tool_name: tool_name.into(),
            args,
        }
    }

    /// Create a timeout error.
    #[must_use]
    pub fn timeout(duration: Duration) -> Self {
        Self::Timeout(duration)
    }

    /// Create a validation failed error.
    #[must_use]
    pub fn validation_failed(msg: impl Into<String>, field: Option<String>) -> Self {
        Self::ValidationFailed {
            message: msg.into(),
            field,
        }
    }

    /// Get the error message.
    #[must_use]
    pub fn message(&self) -> String {
        self.to_string()
    }

    /// Check if this is an approval required error.
    #[must_use]
    pub fn is_approval_required(&self) -> bool {
        matches!(self, Self::ApprovalRequired { .. })
    }

    /// Check if this is a call deferred error.
    #[must_use]
    pub fn is_call_deferred(&self) -> bool {
        matches!(self, Self::CallDeferred { .. })
    }

    /// Check if this is a model retry error.
    #[must_use]
    pub fn is_model_retry(&self) -> bool {
        matches!(self, Self::ModelRetry(_))
    }
}

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        Self::execution_failed(s)
    }
}

impl From<&str> for ToolError {
    fn from(s: &str) -> Self {
        Self::execution_failed(s)
    }
}

/// Serializable error information for tool return.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolErrorInfo {
    /// Error type/code.
    pub error_type: String,
    /// Human-readable message.
    pub message: String,
    /// Whether the error is retryable.
    pub retryable: bool,
    /// Additional details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ToolErrorInfo {
    /// Create a new error info.
    #[must_use]
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error_type: error_type.into(),
            message: message.into(),
            retryable: false,
            details: None,
        }
    }

    /// Set retryable flag.
    #[must_use]
    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    /// Add details.
    #[must_use]
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

impl From<&ToolError> for ToolErrorInfo {
    fn from(err: &ToolError) -> Self {
        let error_type = match err {
            ToolError::ExecutionFailed { .. } => "execution_failed",
            ToolError::InvalidArguments(_) => "invalid_arguments",
            ToolError::NotFound(_) => "not_found",
            ToolError::ModelRetry(_) => "model_retry",
            ToolError::ApprovalRequired { .. } => "approval_required",
            ToolError::CallDeferred { .. } => "call_deferred",
            ToolError::Timeout(_) => "timeout",
            ToolError::ValidationFailed { .. } => "validation_failed",
            ToolError::Cancelled => "cancelled",
            ToolError::ToolReturnedError(_) => "tool_error",
            ToolError::Json(_) => "json_error",
            ToolError::Other(_) => "other",
        };

        Self {
            error_type: error_type.to_string(),
            message: err.message(),
            retryable: err.is_retryable(),
            details: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_failed() {
        let err = ToolError::execution_failed("Something went wrong");
        assert!(!err.is_retryable());
        assert!(err.message().contains("Something went wrong"));
    }

    #[test]
    fn test_retryable_error() {
        let err = ToolError::retryable("Temporary failure");
        assert!(err.is_retryable());
    }

    #[test]
    fn test_not_found() {
        let err = ToolError::not_found("unknown_tool");
        assert!(!err.is_retryable());
        assert!(err.message().contains("unknown_tool"));
    }

    #[test]
    fn test_approval_required() {
        let err = ToolError::approval_required("dangerous_tool", serde_json::json!({"x": 1}));
        assert!(err.is_approval_required());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_call_deferred() {
        let err = ToolError::call_deferred("slow_tool", serde_json::json!({"a": "b"}));
        assert!(err.is_call_deferred());
    }

    #[test]
    fn test_timeout() {
        let err = ToolError::timeout(Duration::from_secs(30));
        assert!(err.is_retryable());
        assert!(err.message().contains("30"));
    }

    #[test]
    fn test_model_retry() {
        let err = ToolError::model_retry("Invalid format");
        assert!(err.is_model_retry());
        assert!(err.is_retryable());
    }

    #[test]
    fn test_error_info_from_error() {
        let err = ToolError::execution_failed("Test error");
        let info = ToolErrorInfo::from(&err);
        assert_eq!(info.error_type, "execution_failed");
        assert!(!info.retryable);
    }

    #[test]
    fn test_from_string() {
        let err: ToolError = "error message".into();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }
}
