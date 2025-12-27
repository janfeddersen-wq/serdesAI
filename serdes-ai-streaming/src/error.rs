//! Streaming errors.

use thiserror::Error;

/// Errors that can occur during streaming.
#[derive(Debug, Error)]
pub enum StreamError {
    /// Model error occurred.
    #[error("Model error: {0}")]
    Model(String),

    /// Parse error for delta.
    #[error("Failed to parse delta: {0}")]
    ParseDelta(String),

    /// Parse error for SSE event.
    #[error("Failed to parse SSE event: {0}")]
    ParseSse(String),

    /// JSON parse error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Stream was interrupted.
    #[error("Stream interrupted")]
    Interrupted,

    /// Timeout waiting for next delta.
    #[error("Timeout waiting for delta")]
    Timeout,

    /// Connection closed unexpectedly.
    #[error("Connection closed unexpectedly")]
    ConnectionClosed,

    /// Connection error.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Send error.
    #[error("Send error: {0}")]
    Send(String),

    /// Receive error.
    #[error("Receive error: {0}")]
    Receive(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid state.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

impl StreamError {
    /// Check if the error is recoverable.
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::Timeout | Self::Interrupted)
    }

    /// Create from any error.
    pub fn from_err<E: std::fmt::Display>(err: E) -> Self {
        Self::Other(err.to_string())
    }
}

/// Result type for streaming operations.
pub type StreamResult<T> = Result<T, StreamError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = StreamError::Timeout;
        assert_eq!(err.to_string(), "Timeout waiting for delta");
    }

    #[test]
    fn test_recoverable() {
        assert!(StreamError::Timeout.is_recoverable());
        assert!(StreamError::Interrupted.is_recoverable());
        assert!(!StreamError::ConnectionClosed.is_recoverable());
    }
}
