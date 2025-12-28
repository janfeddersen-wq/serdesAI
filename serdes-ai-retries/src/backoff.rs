//! Backoff strategies.

use crate::error::RetryableError;
use crate::strategy::RetryStrategy;
use async_trait::async_trait;
use std::time::Duration;

/// Exponential backoff with optional jitter.
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Initial delay.
    pub initial_delay: Duration,
    /// Maximum delay.
    pub max_delay: Duration,
    /// Jitter factor (0.0 to 1.0).
    pub jitter: f64,
    /// Multiplier for each retry.
    pub multiplier: f64,
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            jitter: 0.1,
            multiplier: 2.0,
        }
    }
}

impl ExponentialBackoff {
    /// Create a new exponential backoff.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder.
    #[must_use]
    pub fn builder() -> ExponentialBackoffBuilder {
        ExponentialBackoffBuilder::default()
    }

    /// Calculate delay for an attempt.
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_delay = self.initial_delay.as_secs_f64() * self.multiplier.powi(attempt as i32);
        let jitter = base_delay * self.jitter * rand_jitter();
        let delay = (base_delay + jitter).min(self.max_delay.as_secs_f64());
        Duration::from_secs_f64(delay.max(0.0))
    }
}

#[async_trait]
impl RetryStrategy for ExponentialBackoff {
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> Option<Duration> {
        if attempt > self.max_retries || !error.is_retryable() {
            return None;
        }
        Some(self.calculate_delay(attempt))
    }

    fn max_retries(&self) -> u32 {
        self.max_retries
    }
}

/// Builder for ExponentialBackoff.
#[derive(Debug, Default)]
pub struct ExponentialBackoffBuilder {
    max_retries: Option<u32>,
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
    jitter: Option<f64>,
    multiplier: Option<f64>,
}

impl ExponentialBackoffBuilder {
    /// Set max retries.
    #[must_use]
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Set initial delay.
    #[must_use]
    pub fn initial_delay(mut self, d: Duration) -> Self {
        self.initial_delay = Some(d);
        self
    }

    /// Set max delay.
    #[must_use]
    pub fn max_delay(mut self, d: Duration) -> Self {
        self.max_delay = Some(d);
        self
    }

    /// Set jitter factor.
    #[must_use]
    pub fn jitter(mut self, j: f64) -> Self {
        self.jitter = Some(j);
        self
    }

    /// Set multiplier.
    #[must_use]
    pub fn multiplier(mut self, m: f64) -> Self {
        self.multiplier = Some(m);
        self
    }

    /// Build the backoff strategy.
    #[must_use]
    pub fn build(self) -> ExponentialBackoff {
        let mut backoff = ExponentialBackoff::default();
        if let Some(v) = self.max_retries {
            backoff.max_retries = v;
        }
        if let Some(v) = self.initial_delay {
            backoff.initial_delay = v;
        }
        if let Some(v) = self.max_delay {
            backoff.max_delay = v;
        }
        if let Some(v) = self.jitter {
            backoff.jitter = v;
        }
        if let Some(v) = self.multiplier {
            backoff.multiplier = v;
        }
        backoff
    }
}

/// Fixed delay between retries.
#[derive(Debug, Clone)]
pub struct FixedDelay {
    /// Delay between retries.
    pub delay: Duration,
    /// Maximum retries.
    pub max_retries: u32,
}

impl FixedDelay {
    /// Create a new fixed delay strategy.
    #[must_use]
    pub fn new(delay: Duration, max_retries: u32) -> Self {
        Self { delay, max_retries }
    }
}

#[async_trait]
impl RetryStrategy for FixedDelay {
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> Option<Duration> {
        if attempt > self.max_retries || !error.is_retryable() {
            None
        } else {
            Some(self.delay)
        }
    }

    fn max_retries(&self) -> u32 {
        self.max_retries
    }
}

/// Linear backoff.
#[derive(Debug, Clone)]
pub struct LinearBackoff {
    /// Initial delay.
    pub initial_delay: Duration,
    /// Increment per retry.
    pub increment: Duration,
    /// Maximum delay.
    pub max_delay: Duration,
    /// Maximum retries.
    pub max_retries: u32,
}

impl LinearBackoff {
    /// Create a new linear backoff.
    #[must_use]
    pub fn new(
        initial_delay: Duration,
        increment: Duration,
        max_delay: Duration,
        max_retries: u32,
    ) -> Self {
        Self {
            initial_delay,
            increment,
            max_delay,
            max_retries,
        }
    }

    /// Calculate delay for an attempt.
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let delay = self.initial_delay + self.increment * attempt.saturating_sub(1);
        delay.min(self.max_delay)
    }
}

#[async_trait]
impl RetryStrategy for LinearBackoff {
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> Option<Duration> {
        if attempt > self.max_retries || !error.is_retryable() {
            None
        } else {
            Some(self.calculate_delay(attempt))
        }
    }

    fn max_retries(&self) -> u32 {
        self.max_retries
    }
}

/// Generate a random jitter factor between -1.0 and 1.0.
fn rand_jitter() -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.gen_range(-1.0..1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_backoff_default() {
        let backoff = ExponentialBackoff::new();
        assert_eq!(backoff.max_retries, 3);
        assert_eq!(backoff.initial_delay, Duration::from_millis(100));
        assert_eq!(backoff.multiplier, 2.0);
    }

    #[test]
    fn test_exponential_backoff_builder() {
        let backoff = ExponentialBackoff::builder()
            .max_retries(5)
            .initial_delay(Duration::from_millis(50))
            .max_delay(Duration::from_secs(10))
            .jitter(0.2)
            .build();

        assert_eq!(backoff.max_retries, 5);
        assert_eq!(backoff.initial_delay, Duration::from_millis(50));
        assert_eq!(backoff.max_delay, Duration::from_secs(10));
        assert_eq!(backoff.jitter, 0.2);
    }

    #[test]
    fn test_exponential_backoff_delay() {
        let backoff = ExponentialBackoff::builder()
            .initial_delay(Duration::from_millis(100))
            .multiplier(2.0)
            .jitter(0.0)
            .build();

        // Without jitter, delays should be predictable
        let delay1 = backoff.calculate_delay(1);
        let delay2 = backoff.calculate_delay(2);
        let delay3 = backoff.calculate_delay(3);

        assert_eq!(delay1, Duration::from_millis(200));
        assert_eq!(delay2, Duration::from_millis(400));
        assert_eq!(delay3, Duration::from_millis(800));
    }

    #[test]
    fn test_exponential_backoff_max_delay() {
        let backoff = ExponentialBackoff::builder()
            .initial_delay(Duration::from_secs(1))
            .max_delay(Duration::from_secs(5))
            .multiplier(10.0)
            .jitter(0.0)
            .build();

        // Even with large multiplier, should cap at max
        let delay = backoff.calculate_delay(5);
        assert!(delay <= Duration::from_secs(5));
    }

    #[test]
    fn test_exponential_backoff_should_retry() {
        let backoff = ExponentialBackoff::builder().max_retries(3).build();

        let error = RetryableError::http(500, "error");
        assert!(backoff.should_retry(&error, 1).is_some());
        assert!(backoff.should_retry(&error, 3).is_some());
        assert!(backoff.should_retry(&error, 4).is_none());

        // Non-retryable error
        let error = RetryableError::http(400, "bad request");
        assert!(backoff.should_retry(&error, 1).is_none());
    }

    #[test]
    fn test_fixed_delay() {
        let delay = FixedDelay::new(Duration::from_secs(1), 3);

        let error = RetryableError::http(500, "error");
        assert_eq!(
            delay.should_retry(&error, 1),
            Some(Duration::from_secs(1))
        );
        assert_eq!(
            delay.should_retry(&error, 3),
            Some(Duration::from_secs(1))
        );
        assert_eq!(delay.should_retry(&error, 4), None);
    }

    #[test]
    fn test_linear_backoff() {
        let backoff = LinearBackoff::new(
            Duration::from_millis(100),
            Duration::from_millis(100),
            Duration::from_secs(1),
            5,
        );

        assert_eq!(backoff.calculate_delay(1), Duration::from_millis(100));
        assert_eq!(backoff.calculate_delay(2), Duration::from_millis(200));
        assert_eq!(backoff.calculate_delay(3), Duration::from_millis(300));

        // Check max delay cap
        assert_eq!(backoff.calculate_delay(20), Duration::from_secs(1));
    }
}
