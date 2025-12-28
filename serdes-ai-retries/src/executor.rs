//! Retry executor for running operations with retries.

use crate::config::RetryConfig;
use crate::error::{RetryResult, RetryableError};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// State of a retry attempt.
#[derive(Debug, Clone)]
pub struct RetryState {
    /// Current attempt number (1-indexed).
    pub attempt: u32,
    /// Last error message.
    pub last_error: Option<String>,
    /// Total time spent waiting.
    pub total_wait_time: Duration,
    /// History of attempts.
    pub history: Vec<AttemptInfo>,
}

impl Default for RetryState {
    fn default() -> Self {
        Self {
            attempt: 0,
            last_error: None,
            total_wait_time: Duration::ZERO,
            history: Vec::new(),
        }
    }
}

/// Information about a single attempt.
#[derive(Debug, Clone)]
pub struct AttemptInfo {
    /// Attempt number.
    pub attempt: u32,
    /// Whether it succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Time waited before this attempt.
    pub wait_time: Duration,
}

/// Execute an operation with retries.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_retries::{with_retry, RetryConfig};
///
/// let config = RetryConfig::for_api();
/// let result = with_retry(&config, || async {
///     // Your async operation here
///     Ok("success")
/// }).await?;
/// ```
pub async fn with_retry<F, Fut, T>(config: &RetryConfig, operation: F) -> RetryResult<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = RetryResult<T>>,
{
    let mut state = RetryState::default();
    let max_attempts = config.max_retries.saturating_add(1);

    loop {
        state.attempt += 1;

        debug!(
            attempt = state.attempt,
            max_attempts,
            max_retries = config.max_retries,
            "Executing retry attempt"
        );

        match operation().await {
            Ok(result) => {
                state.history.push(AttemptInfo {
                    attempt: state.attempt,
                    success: true,
                    error: None,
                    wait_time: Duration::ZERO,
                });
                return Ok(result);
            }
            Err(error) => {
                let should_retry = state.attempt < max_attempts
                    && config.retry_on.should_retry(&error);

                if !should_retry {
                    warn!(
                        attempt = state.attempt,
                        error = %error,
                        "Retry exhausted or error not retryable"
                    );
                    return Err(error);
                }

                let wait = config.wait.calculate(state.attempt, error.retry_after());
                state.total_wait_time += wait;
                state.last_error = Some(format!("{}", error));

                state.history.push(AttemptInfo {
                    attempt: state.attempt,
                    success: false,
                    error: Some(format!("{}", error)),
                    wait_time: wait,
                });

                debug!(
                    attempt = state.attempt,
                    wait_ms = wait.as_millis(),
                    error = %error,
                    "Waiting before retry"
                );

                sleep(wait).await;
            }
        }
    }
}

/// Execute with retries and get state information.
pub async fn with_retry_state<F, Fut, T>(
    config: &RetryConfig,
    operation: F,
) -> (RetryResult<T>, RetryState)
where
    F: Fn() -> Fut,
    Fut: Future<Output = RetryResult<T>>,
{
    let mut state = RetryState::default();
    let max_attempts = config.max_retries.saturating_add(1);

    loop {
        state.attempt += 1;

        match operation().await {
            Ok(result) => {
                state.history.push(AttemptInfo {
                    attempt: state.attempt,
                    success: true,
                    error: None,
                    wait_time: Duration::ZERO,
                });
                return (Ok(result), state);
            }
            Err(error) => {
                let should_retry = state.attempt < max_attempts
                    && config.retry_on.should_retry(&error);

                if !should_retry {
                    return (Err(error), state);
                }

                let wait = config.wait.calculate(state.attempt, error.retry_after());
                state.total_wait_time += wait;
                state.last_error = Some(format!("{}", error));

                state.history.push(AttemptInfo {
                    attempt: state.attempt,
                    success: false,
                    error: Some(format!("{}", error)),
                    wait_time: wait,
                });

                sleep(wait).await;
            }
        }
    }
}

/// Builder for retry operations.
pub struct Retry<'a> {
    config: &'a RetryConfig,
}

impl<'a> Retry<'a> {
    /// Create a new retry builder.
    pub fn new(config: &'a RetryConfig) -> Self {
        Self { config }
    }

    /// Run the operation with retries.
    pub async fn run<F, Fut, T>(self, operation: F) -> RetryResult<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = RetryResult<T>>,
    {
        with_retry(self.config, operation).await
    }

    /// Run and get state.
    pub async fn run_with_state<F, Fut, T>(self, operation: F) -> (RetryResult<T>, RetryState)
    where
        F: Fn() -> Fut,
        Fut: Future<Output = RetryResult<T>>,
    {
        with_retry_state(self.config, operation).await
    }
}

/// Wrap a result type for retry compatibility.
pub trait IntoRetryable<T> {
    /// Convert into a retryable result.
    fn into_retryable(self) -> RetryResult<T>;
}

impl<T, E: Into<RetryableError>> IntoRetryable<T> for Result<T, E> {
    fn into_retryable(self) -> RetryResult<T> {
        self.map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_with_retry_immediate_success() {
        let config = RetryConfig::new().max_retries(3);
        let result = with_retry(&config, || async { Ok::<_, RetryableError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_retry_eventual_success() {
        let config = RetryConfig::new()
            .max_retries(3)
            .fixed(Duration::from_millis(1));

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result = with_retry(&config, || {
            let attempts = attempts_clone.clone();
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(RetryableError::http(500, "server error"))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_exhausted() {
        let config = RetryConfig::new()
            .max_retries(2)
            .fixed(Duration::from_millis(1));

        let result = with_retry(&config, || async {
            Err::<i32, _>(RetryableError::http(500, "always fails"))
        })
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_with_retry_non_retryable() {
        let config = RetryConfig::new().max_retries(3);

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result = with_retry(&config, || {
            let attempts = attempts_clone.clone();
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(RetryableError::http(400, "bad request"))
            }
        })
        .await;

        assert!(result.is_err());
        // Should only try once since 400 is not retryable
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_state() {
        let config = RetryConfig::new()
            .max_retries(3)
            .fixed(Duration::from_millis(1));

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let (result, state) = with_retry_state(&config, || {
            let attempts = attempts_clone.clone();
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err(RetryableError::http(500, "error"))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(state.attempt, 2);
        assert_eq!(state.history.len(), 2);
        assert!(!state.history[0].success);
        assert!(state.history[1].success);
    }

    #[tokio::test]
    async fn test_retry_builder() {
        let config = RetryConfig::new();
        let result = Retry::new(&config)
            .run(|| async { Ok::<_, RetryableError>("success") })
            .await;

        assert_eq!(result.unwrap(), "success");
    }
}
