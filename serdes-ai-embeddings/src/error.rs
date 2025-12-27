//! Embedding errors.

use thiserror::Error;

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// HTTP error from provider.
    #[error("HTTP error {status}: {body}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },

    /// API error from provider.
    #[error("API error: {0}")]
    Api(String),

    /// Rate limited by provider.
    #[error("Rate limited: retry after {retry_after:?}")]
    RateLimited {
        /// Suggested retry time.
        retry_after: Option<std::time::Duration>,
    },

    /// Input too long.
    #[error("Input too long: {length} tokens (max: {max})")]
    InputTooLong {
        /// Actual length.
        length: usize,
        /// Maximum allowed.
        max: usize,
    },

    /// Feature not supported by this model.
    #[error("Not supported: {0}")]
    NotSupported(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Network/IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Other error.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl EmbeddingError {
    /// Create an API error.
    pub fn api(msg: impl Into<String>) -> Self {
        Self::Api(msg.into())
    }

    /// Create a config error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. }
                | Self::Http { status: 429 | 500..=599, .. }
        )
    }

    /// Get suggested retry delay.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            Self::RateLimited { retry_after } => *retry_after,
            _ => None,
        }
    }
}

/// Result type for embedding operations.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EmbeddingError::api("invalid key");
        assert!(err.to_string().contains("invalid key"));
    }

    #[test]
    fn test_retryable() {
        assert!(EmbeddingError::RateLimited { retry_after: None }.is_retryable());
        assert!(EmbeddingError::Http { status: 500, body: String::new() }.is_retryable());
        assert!(!EmbeddingError::Api("bad".into()).is_retryable());
    }
}
