//! Retry strategy trait.

use crate::error::RetryableError;
use async_trait::async_trait;
use std::time::Duration;

/// Trait for retry strategies.
#[async_trait]
pub trait RetryStrategy: Send + Sync {
    /// Determine if and how long to wait before retrying.
    ///
    /// Returns `Some(duration)` if the operation should be retried after waiting,
    /// or `None` if no more retries should be attempted.
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> Option<Duration>;

    /// Get the maximum number of retries.
    fn max_retries(&self) -> u32;

    /// Check if retries are exhausted.
    fn is_exhausted(&self, attempt: u32) -> bool {
        attempt >= self.max_retries()
    }
}

/// Strategy that never retries.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRetry;

impl NoRetry {
    /// Create a new no-retry strategy.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RetryStrategy for NoRetry {
    fn should_retry(&self, _error: &RetryableError, _attempt: u32) -> Option<Duration> {
        None
    }

    fn max_retries(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_retry() {
        let strategy = NoRetry::new();
        let error = RetryableError::http(500, "error");

        assert!(strategy.should_retry(&error, 0).is_none());
        assert_eq!(strategy.max_retries(), 0);
        assert!(strategy.is_exhausted(0));
    }
}
