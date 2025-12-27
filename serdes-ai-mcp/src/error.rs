//! MCP error types.

use crate::types::JsonRpcError;
use thiserror::Error;

/// MCP errors.
#[derive(Debug, Error)]
pub enum McpError {
    /// Transport error.
    #[error("Transport error: {0}")]
    Transport(String),

    /// Protocol error.
    #[error("Protocol error {code}: {message}")]
    Protocol {
        /// Error code.
        code: i32,
        /// Error message.
        message: String,
    },

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP error.
    #[error("HTTP error: {0}")]
    Http(u16),

    /// Connection closed.
    #[error("Connection closed")]
    ConnectionClosed,

    /// Not initialized.
    #[error("Client not initialized")]
    NotInitialized,

    /// No result in response.
    #[error("No result in response")]
    NoResult,

    /// Tool not found.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Resource not found.
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Timeout.
    #[error("Timeout")]
    Timeout,

    /// Other error.
    #[error("{0}")]
    Other(String),
}

impl From<JsonRpcError> for McpError {
    fn from(err: JsonRpcError) -> Self {
        Self::Protocol {
            code: err.code,
            message: err.message,
        }
    }
}

impl McpError {
    /// Create from any error.
    pub fn from_err<E: std::fmt::Display>(err: E) -> Self {
        Self::Other(err.to_string())
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::Timeout | Self::Transport(_))
    }
}

/// Result type for MCP operations.
pub type McpResult<T> = Result<T, McpError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = McpError::Transport("connection failed".to_string());
        assert!(err.to_string().contains("connection failed"));
    }

    #[test]
    fn test_from_json_rpc_error() {
        let rpc_err = JsonRpcError {
            code: -32600,
            message: "Invalid request".to_string(),
            data: None,
        };
        let err: McpError = rpc_err.into();
        assert!(matches!(err, McpError::Protocol { code: -32600, .. }));
    }

    #[test]
    fn test_recoverable() {
        assert!(McpError::Timeout.is_recoverable());
        assert!(!McpError::NotInitialized.is_recoverable());
    }
}
