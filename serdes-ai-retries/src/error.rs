//! Retry error types.

use std::time::Duration;
use thiserror::Error;

/// Errors that can be retried.
#[derive(Debug, Error)]
pub enum RetryableError {
    /// HTTP error with status code.
    #[error("HTTP error {status}: {body}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
        /// Retry-After header value.
        retry_after: Option<Duration>,
    },

    /// Rate limited.
    #[error("Rate limited")]
    RateLimited {
        /// Suggested retry time.
        retry_after: Option<Duration>,
    },

    /// Timeout.
    #[error("Timeout")]
    Timeout,

    /// Connection error.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Other error.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl RetryableError {
    /// Create an HTTP error.
    pub fn http(status: u16, body: impl Into<String>) -> Self {
        Self::Http {
            status,
            body: body.into(),
            retry_after: None,
        }
    }

    /// Create a rate limit error.
    pub fn rate_limited(retry_after: Option<Duration>) -> Self {
        Self::RateLimited { retry_after }
    }

    /// Create a connection error.
    pub fn connection(msg: impl Into<String>) -> Self {
        Self::Connection(msg.into())
    }

    /// Get the suggested retry-after duration.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Http { retry_after, .. } => *retry_after,
            Self::RateLimited { retry_after } => *retry_after,
            _ => None,
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http { status, .. } => {
                *status == 429 || (500..=599).contains(status)
            }
            Self::RateLimited { .. } => true,
            Self::Timeout => true,
            Self::Connection(_) => true,
            Self::Other(_) => false,
        }
    }

    /// Get the HTTP status if this is an HTTP error.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } => Some(*status),
            Self::RateLimited { .. } => Some(429),
            _ => None,
        }
    }
}

/// Result type for retry operations.
pub type RetryResult<T> = Result<T, RetryableError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retryable_errors() {
        assert!(RetryableError::rate_limited(None).is_retryable());
        assert!(RetryableError::Timeout.is_retryable());
        assert!(RetryableError::http(500, "error").is_retryable());
        assert!(RetryableError::http(429, "rate limited").is_retryable());
        assert!(!RetryableError::http(400, "bad request").is_retryable());
    }

    #[test]
    fn test_retry_after() {
        let err = RetryableError::rate_limited(Some(Duration::from_secs(5)));
        assert_eq!(err.retry_after(), Some(Duration::from_secs(5)));

        let err = RetryableError::Timeout;
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn test_status() {
        let err = RetryableError::http(503, "unavailable");
        assert_eq!(err.status(), Some(503));

        let err = RetryableError::RateLimited { retry_after: None };
        assert_eq!(err.status(), Some(429));
    }
}
