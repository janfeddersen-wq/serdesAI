//! Tool-specific error types.
//!
//! This module provides comprehensive error types for tool execution,
//! including retryable errors, approval flows, and validation failures.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Structured validation error for tool arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationError {
    /// Field that failed validation, if applicable.
    pub field: Option<String>,
    /// Validation error message.
    pub message: String,
}

impl ValidationError {
    /// Create a new validation error.
    #[must_use]
    pub fn new(field: Option<String>, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

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

    /// Tool argument validation failed.
    #[error("Tool argument validation failed for '{tool_name}'")]
    ValidationFailed {
        /// Name of the tool.
        tool_name: String,
        /// Validation errors.
        errors: Vec<ValidationError>,
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

    /// Create a validation failed error with multiple issues.
    #[must_use]
    pub fn validation_failed(tool_name: impl Into<String>, errors: Vec<ValidationError>) -> Self {
        Self::ValidationFailed {
            tool_name: tool_name.into(),
            errors,
        }
    }

    /// Create a validation failed error with a single issue.
    #[must_use]
    pub fn validation_error(
        tool_name: impl Into<String>,
        field: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::validation_failed(tool_name, vec![ValidationError::new(field, message)])
    }

    /// Create an invalid arguments error for argument parsing.
    #[must_use]
    pub fn invalid_arguments(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::validation_failed(tool_name, vec![ValidationError::new(None, message)])
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

        let details = match err {
            ToolError::ValidationFailed { errors, .. } => serde_json::to_value(errors).ok(),
            _ => None,
        };

        Self {
            error_type: error_type.to_string(),
            message: err.message(),
            retryable: err.is_retryable(),
            details,
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
    fn test_validation_failed() {
        let err =
            ToolError::validation_error("test_tool", Some("field".to_string()), "Invalid value");
        assert!(!err.is_retryable());
        let info = ToolErrorInfo::from(&err);
        assert_eq!(info.error_type, "validation_failed");
        assert!(info.details.is_some());
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
