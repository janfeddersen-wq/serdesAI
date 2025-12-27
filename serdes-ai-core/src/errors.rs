//! Error types for serdes-ai.
//!
//! This module provides a comprehensive error hierarchy that matches pydantic-ai's
//! error system, enabling proper error handling and retry logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// The main error type for serdes-ai operations.
#[derive(Error, Debug)]
pub enum SerdesAiError {
    /// Error during agent execution.
    #[error(transparent)]
    AgentRun(#[from] AgentRunError),

    /// Model requested a retry.
    #[error(transparent)]
    ModelRetry(#[from] ModelRetry),

    /// API error from the model provider.
    #[error(transparent)]
    ModelAPI(#[from] ModelAPIError),

    /// HTTP-level error.
    #[error(transparent)]
    ModelHTTP(#[from] ModelHTTPError),

    /// User-defined error from tools or validators.
    #[error(transparent)]
    User(#[from] UserError),

    /// Usage limits exceeded.
    #[error(transparent)]
    UsageLimit(#[from] UsageLimitExceeded),

    /// Unexpected model behavior.
    #[error(transparent)]
    UnexpectedBehavior(#[from] UnexpectedModelBehavior),

    /// Tool requested a retry.
    #[error(transparent)]
    ToolRetry(#[from] ToolRetryError),

    /// Tool requires user approval.
    #[error(transparent)]
    ApprovalRequired(#[from] ApprovalRequired),

    /// Tool call was deferred.
    #[error(transparent)]
    CallDeferred(#[from] CallDeferred),

    /// Incomplete tool call from model.
    #[error(transparent)]
    IncompleteToolCall(#[from] IncompleteToolCall),

    /// Multiple errors occurred.
    #[error(transparent)]
    FallbackGroup(#[from] FallbackExceptionGroup),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias using SerdesAiError.
pub type Result<T> = std::result::Result<T, SerdesAiError>;

/// Error during agent run execution.
#[derive(Error, Debug, Clone)]
pub struct AgentRunError {
    /// Error message.
    pub message: String,
    /// The run ID where the error occurred.
    pub run_id: Option<String>,
    /// Number of attempts made.
    pub attempts: u32,
    /// Original cause (serialized).
    pub cause: Option<String>,
}

impl fmt::Display for AgentRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent run error: {}", self.message)?;
        if let Some(ref run_id) = self.run_id {
            write!(f, " (run_id: {})", run_id)?;
        }
        if self.attempts > 1 {
            write!(f, " after {} attempts", self.attempts)?;
        }
        Ok(())
    }
}

impl AgentRunError {
    /// Create a new agent run error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            run_id: None,
            attempts: 1,
            cause: None,
        }
    }

    /// Set the run ID.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Set the number of attempts.
    pub fn with_attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts;
        self
    }

    /// Set the cause.
    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }
}

/// Model requested a retry, typically due to validation failure.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct ModelRetry {
    /// Message explaining why retry is needed.
    pub message: String,
}

impl fmt::Display for ModelRetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model retry requested: {}", self.message)
    }
}

impl ModelRetry {
    /// Create a new model retry error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// API error from the model provider.
#[derive(Error, Debug, Clone)]
pub struct ModelAPIError {
    /// HTTP status code.
    pub status_code: u16,
    /// Response body.
    pub body: String,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Error message from the API.
    pub message: Option<String>,
    /// Error code from the API.
    pub error_code: Option<String>,
    /// Whether this error is retryable.
    pub retryable: bool,
    /// Retry-after header value in seconds.
    pub retry_after: Option<u64>,
}

impl fmt::Display for ModelAPIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model API error (status {})", self.status_code)?;
        if let Some(ref msg) = self.message {
            write!(f, ": {}", msg)?;
        }
        if let Some(ref code) = self.error_code {
            write!(f, " [{}]", code)?;
        }
        Ok(())
    }
}

impl ModelAPIError {
    /// Create a new API error.
    pub fn new(status_code: u16, body: impl Into<String>) -> Self {
        Self {
            status_code,
            body: body.into(),
            headers: HashMap::new(),
            message: None,
            error_code: None,
            retryable: status_code == 429 || status_code >= 500,
            retry_after: None,
        }
    }

    /// Set the message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Set the error code.
    pub fn with_error_code(mut self, code: impl Into<String>) -> Self {
        self.error_code = Some(code.into());
        self
    }

    /// Set headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        // Parse retry-after if present
        if let Some(retry_after) = self.headers.get("retry-after") {
            self.retry_after = retry_after.parse().ok();
        }
        self
    }

    /// Check if this is a rate limit error.
    pub fn is_rate_limit(&self) -> bool {
        self.status_code == 429
    }

    /// Check if this is a server error.
    pub fn is_server_error(&self) -> bool {
        self.status_code >= 500
    }
}

/// HTTP-level error (connection, timeout, etc.).
#[derive(Error, Debug, Clone)]
pub struct ModelHTTPError {
    /// Error message.
    pub message: String,
    /// Whether this is a timeout.
    pub is_timeout: bool,
    /// Whether this is a connection error.
    pub is_connection_error: bool,
    /// URL that was being accessed.
    pub url: Option<String>,
}

impl fmt::Display for ModelHTTPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_timeout {
            write!(f, "Request timeout: {}", self.message)
        } else if self.is_connection_error {
            write!(f, "Connection error: {}", self.message)
        } else {
            write!(f, "HTTP error: {}", self.message)
        }
    }
}

impl ModelHTTPError {
    /// Create a new HTTP error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            is_timeout: false,
            is_connection_error: false,
            url: None,
        }
    }

    /// Create a timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            is_timeout: true,
            is_connection_error: false,
            url: None,
        }
    }

    /// Create a connection error.
    pub fn connection(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            is_timeout: false,
            is_connection_error: true,
            url: None,
        }
    }

    /// Set the URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

/// User-defined error from tools or validators.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct UserError {
    /// Error message.
    pub message: String,
    /// Optional details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl fmt::Display for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "User error: {}", self.message)
    }
}

impl UserError {
    /// Create a new user error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: None,
        }
    }

    /// Add details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Type of usage limit that was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageLimitType {
    /// Request tokens limit.
    RequestTokens,
    /// Response tokens limit.
    ResponseTokens,
    /// Total tokens limit.
    TotalTokens,
    /// Number of requests limit.
    Requests,
}

impl fmt::Display for UsageLimitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestTokens => write!(f, "request_tokens"),
            Self::ResponseTokens => write!(f, "response_tokens"),
            Self::TotalTokens => write!(f, "total_tokens"),
            Self::Requests => write!(f, "requests"),
        }
    }
}

/// Usage limits exceeded.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct UsageLimitExceeded {
    /// Type of limit exceeded.
    pub limit_type: UsageLimitType,
    /// Current value.
    pub current: u64,
    /// Maximum allowed value.
    pub max: u64,
    /// Additional message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl fmt::Display for UsageLimitExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Usage limit exceeded: {} is {} but max is {}",
            self.limit_type, self.current, self.max
        )
    }
}

impl UsageLimitExceeded {
    /// Create a new usage limit error.
    pub fn new(limit_type: UsageLimitType, current: u64, max: u64) -> Self {
        Self {
            limit_type,
            current,
            max,
            message: None,
        }
    }

    /// Add a message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

/// Unexpected behavior from the model.
#[derive(Error, Debug, Clone)]
pub struct UnexpectedModelBehavior {
    /// Description of the unexpected behavior.
    pub message: String,
    /// The response that caused the issue.
    pub response: Option<String>,
}

impl fmt::Display for UnexpectedModelBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unexpected model behavior: {}", self.message)
    }
}

impl UnexpectedModelBehavior {
    /// Create a new unexpected behavior error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            response: None,
        }
    }

    /// Add the response.
    pub fn with_response(mut self, response: impl Into<String>) -> Self {
        self.response = Some(response.into());
        self
    }
}

/// Tool requested a retry.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct ToolRetryError {
    /// Tool name.
    pub tool_name: String,
    /// Retry message.
    pub message: String,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl fmt::Display for ToolRetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tool '{}' retry: {}", self.tool_name, self.message)
    }
}

impl ToolRetryError {
    /// Create a new tool retry error.
    pub fn new(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            message: message.into(),
            tool_call_id: None,
        }
    }

    /// Set the tool call ID.
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

/// Tool requires user approval before execution.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequired {
    /// Tool name.
    pub tool_name: String,
    /// Tool arguments.
    pub args: serde_json::Value,
    /// Reason approval is required.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl fmt::Display for ApprovalRequired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Approval required for tool '{}'", self.tool_name)?;
        if let Some(ref reason) = self.reason {
            write!(f, ": {}", reason)?;
        }
        Ok(())
    }
}

impl ApprovalRequired {
    /// Create a new approval required error.
    pub fn new(tool_name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            args,
            reason: None,
            tool_call_id: None,
        }
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the tool call ID.
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

/// Tool call was deferred for later execution.
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct CallDeferred {
    /// Tool name.
    pub tool_name: String,
    /// Tool arguments.
    pub args: serde_json::Value,
    /// Reason for deferral.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl fmt::Display for CallDeferred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tool call '{}' deferred", self.tool_name)?;
        if let Some(ref reason) = self.reason {
            write!(f, ": {}", reason)?;
        }
        Ok(())
    }
}

impl CallDeferred {
    /// Create a new call deferred error.
    pub fn new(tool_name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            args,
            reason: None,
            tool_call_id: None,
        }
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the tool call ID.
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }
}

/// Incomplete tool call from model (missing required fields).
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteToolCall {
    /// Tool name (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// What's missing.
    pub message: String,
    /// The partial data received.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_data: Option<serde_json::Value>,
}

impl fmt::Display for IncompleteToolCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Incomplete tool call")?;
        if let Some(ref name) = self.tool_name {
            write!(f, " for '{}'", name)?;
        }
        write!(f, ": {}", self.message)
    }
}

impl IncompleteToolCall {
    /// Create a new incomplete tool call error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            tool_name: None,
            message: message.into(),
            partial_data: None,
        }
    }

    /// Set the tool name.
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// Set partial data.
    pub fn with_partial_data(mut self, data: serde_json::Value) -> Self {
        self.partial_data = Some(data);
        self
    }
}

/// Group of errors from fallback attempts.
#[derive(Error, Debug, Clone)]
pub struct FallbackExceptionGroup {
    /// Message describing the group.
    pub message: String,
    /// Individual errors.
    pub errors: Vec<String>,
}

impl fmt::Display for FallbackExceptionGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({} errors)", self.message, self.errors.len())
    }
}

impl FallbackExceptionGroup {
    /// Create a new fallback exception group.
    pub fn new(message: impl Into<String>, errors: Vec<String>) -> Self {
        Self {
            message: message.into(),
            errors,
        }
    }

    /// Create from a list of errors.
    pub fn from_errors<E: std::error::Error>(message: impl Into<String>, errors: Vec<E>) -> Self {
        Self {
            message: message.into(),
            errors: errors.iter().map(|e| e.to_string()).collect(),
        }
    }

    /// Check if this group is empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the number of errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_retry() {
        let err = ModelRetry::new("Invalid JSON output");
        assert_eq!(err.message, "Invalid JSON output");
        assert!(err.to_string().contains("Invalid JSON output"));
    }

    #[test]
    fn test_usage_limit_exceeded() {
        let err = UsageLimitExceeded::new(UsageLimitType::TotalTokens, 5000, 4000);
        assert_eq!(err.current, 5000);
        assert_eq!(err.max, 4000);
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_api_error_is_rate_limit() {
        let err = ModelAPIError::new(429, "Rate limited");
        assert!(err.is_rate_limit());
        assert!(err.retryable);
    }

    #[test]
    fn test_fallback_group() {
        let group = FallbackExceptionGroup::new(
            "All providers failed",
            vec!["Error 1".to_string(), "Error 2".to_string()],
        );
        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());
    }
}
