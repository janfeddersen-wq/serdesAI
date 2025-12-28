//! Retry configuration.

use crate::error::RetryableError;
use std::time::Duration;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Wait strategy.
    pub wait: WaitStrategy,
    /// Retry condition.
    pub retry_on: RetryCondition,
    /// Whether to reraise the last error if all retries fail.
    pub reraise: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            wait: WaitStrategy::ExponentialBackoff {
                initial: Duration::from_millis(500),
                max: Duration::from_secs(60),
                multiplier: 2.0,
            },
            retry_on: RetryCondition::default(),
            reraise: true,
        }
    }
}

impl RetryConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max retries.
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Set the wait strategy.
    pub fn wait(mut self, strategy: WaitStrategy) -> Self {
        self.wait = strategy;
        self
    }

    /// Use exponential backoff.
    pub fn exponential(mut self, initial: Duration, max: Duration) -> Self {
        self.wait = WaitStrategy::ExponentialBackoff {
            initial,
            max,
            multiplier: 2.0,
        };
        self
    }

    /// Use exponential backoff with jitter.
    pub fn exponential_jitter(
        mut self,
        initial: Duration,
        max: Duration,
        jitter: f64,
    ) -> Self {
        self.wait = WaitStrategy::ExponentialJitter {
            initial,
            max,
            multiplier: 2.0,
            jitter,
        };
        self
    }

    /// Use fixed delay.
    pub fn fixed(mut self, delay: Duration) -> Self {
        self.wait = WaitStrategy::Fixed(delay);
        self
    }

    /// Use linear backoff.
    pub fn linear(mut self, initial: Duration, increment: Duration, max: Duration) -> Self {
        self.wait = WaitStrategy::Linear {
            initial,
            increment,
            max,
        };
        self
    }

    /// Set retry condition.
    pub fn retry_on(mut self, condition: RetryCondition) -> Self {
        self.retry_on = condition;
        self
    }

    /// Set whether to reraise the last error.
    pub fn reraise(mut self, reraise: bool) -> Self {
        self.reraise = reraise;
        self
    }

    /// Create config for API calls with sensible defaults.
    pub fn for_api() -> Self {
        Self::new()
            .max_retries(3)
            .exponential_jitter(
                Duration::from_millis(500),
                Duration::from_secs(60),
                0.1,
            )
            .retry_on(
                RetryCondition::new()
                    .on_rate_limit()
                    .on_server_errors(),
            )
    }

    /// Create config that never retries.
    pub fn no_retry() -> Self {
        Self::new().max_retries(0)
    }
}

/// Strategy for waiting between retries.
#[derive(Debug, Clone)]
pub enum WaitStrategy {
    /// No waiting.
    None,
    /// Fixed delay.
    Fixed(Duration),
    /// Exponential backoff.
    ExponentialBackoff {
        /// Initial delay.
        initial: Duration,
        /// Maximum delay.
        max: Duration,
        /// Multiplier for each attempt.
        multiplier: f64,
    },
    /// Exponential backoff with jitter.
    ExponentialJitter {
        /// Initial delay.
        initial: Duration,
        /// Maximum delay.
        max: Duration,
        /// Multiplier for each attempt.
        multiplier: f64,
        /// Jitter factor (0.0 to 1.0).
        jitter: f64,
    },
    /// Linear backoff.
    Linear {
        /// Initial delay.
        initial: Duration,
        /// Increment per attempt.
        increment: Duration,
        /// Maximum delay.
        max: Duration,
    },
    /// Respect Retry-After header.
    RetryAfter {
        /// Fallback if no header.
        fallback: Box<WaitStrategy>,
        /// Maximum wait time.
        max_wait: Duration,
    },
}

impl WaitStrategy {
    /// Calculate the wait duration for a given attempt.
    pub fn calculate(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        match self {
            WaitStrategy::None => Duration::ZERO,
            WaitStrategy::Fixed(d) => *d,
            WaitStrategy::ExponentialBackoff {
                initial,
                max,
                multiplier,
            } => {
                let delay = initial.as_secs_f64() * multiplier.powi(attempt as i32 - 1);
                Duration::from_secs_f64(delay.min(max.as_secs_f64()))
            }
            WaitStrategy::ExponentialJitter {
                initial,
                max,
                multiplier,
                jitter,
            } => {
                let base = initial.as_secs_f64() * multiplier.powi(attempt as i32 - 1);
                let jitter_amount = base * jitter * random_jitter();
                let delay = (base + jitter_amount).min(max.as_secs_f64());
                Duration::from_secs_f64(delay)
            }
            WaitStrategy::Linear {
                initial,
                increment,
                max,
            } => {
                let delay = *initial + *increment * (attempt - 1);
                delay.min(*max)
            }
            WaitStrategy::RetryAfter { fallback, max_wait } => retry_after
                .map(|d| d.min(*max_wait))
                .unwrap_or_else(|| fallback.calculate(attempt, None)),
        }
    }
}

/// Condition for retrying.
#[derive(Debug, Clone, Default)]
pub struct RetryCondition {
    /// HTTP status codes to retry on.
    pub on_status_codes: Vec<u16>,
    /// Custom predicate function.
    pub custom: Option<fn(&RetryableError) -> bool>,
}

impl RetryCondition {
    /// Create a new empty condition.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add status codes to retry on.
    pub fn on_status(mut self, codes: impl IntoIterator<Item = u16>) -> Self {
        self.on_status_codes.extend(codes);
        self
    }

    /// Retry on server errors (5xx).
    pub fn on_server_errors(mut self) -> Self {
        self.on_status_codes.extend(500..=599);
        self
    }

    /// Retry on rate limit (429).
    pub fn on_rate_limit(mut self) -> Self {
        self.on_status_codes.push(429);
        self
    }

    /// Retry on timeout errors.
    pub fn on_timeout(self) -> Self {
        // Timeout is always retryable by default
        self
    }

    /// Set a custom predicate.
    pub fn with_custom(mut self, predicate: fn(&RetryableError) -> bool) -> Self {
        self.custom = Some(predicate);
        self
    }

    /// Check if an error should be retried.
    pub fn should_retry(&self, error: &RetryableError) -> bool {
        // Check custom predicate first
        if let Some(predicate) = self.custom {
            return predicate(error);
        }

        // Check status codes
        if let Some(status) = error.status() {
            if self.on_status_codes.contains(&status) {
                return true;
            }
        }

        // Default to error's own retryable check
        error.is_retryable()
    }
}

/// Generate a random jitter factor between -1.0 and 1.0.
fn random_jitter() -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.gen_range(-1.0..1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(config.reraise);
    }

    #[test]
    fn test_config_builder() {
        let config = RetryConfig::new()
            .max_retries(5)
            .fixed(Duration::from_secs(1))
            .reraise(false);

        assert_eq!(config.max_retries, 5);
        assert!(!config.reraise);
    }

    #[test]
    fn test_wait_strategy_fixed() {
        let strategy = WaitStrategy::Fixed(Duration::from_secs(1));
        assert_eq!(strategy.calculate(1, None), Duration::from_secs(1));
        assert_eq!(strategy.calculate(3, None), Duration::from_secs(1));
    }

    #[test]
    fn test_wait_strategy_exponential() {
        let strategy = WaitStrategy::ExponentialBackoff {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(10),
            multiplier: 2.0,
        };

        assert_eq!(strategy.calculate(1, None), Duration::from_millis(100));
        assert_eq!(strategy.calculate(2, None), Duration::from_millis(200));
        assert_eq!(strategy.calculate(3, None), Duration::from_millis(400));
    }

    #[test]
    fn test_wait_strategy_linear() {
        let strategy = WaitStrategy::Linear {
            initial: Duration::from_millis(100),
            increment: Duration::from_millis(100),
            max: Duration::from_secs(10),
        };

        assert_eq!(strategy.calculate(1, None), Duration::from_millis(100));
        assert_eq!(strategy.calculate(2, None), Duration::from_millis(200));
        assert_eq!(strategy.calculate(3, None), Duration::from_millis(300));
    }

    #[test]
    fn test_wait_strategy_retry_after() {
        let strategy = WaitStrategy::RetryAfter {
            fallback: Box::new(WaitStrategy::Fixed(Duration::from_secs(1))),
            max_wait: Duration::from_secs(60),
        };

        // With retry-after header
        assert_eq!(
            strategy.calculate(1, Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );

        // Without retry-after header (uses fallback)
        assert_eq!(strategy.calculate(1, None), Duration::from_secs(1));
    }

    #[test]
    fn test_retry_condition() {
        let condition = RetryCondition::new()
            .on_rate_limit()
            .on_server_errors();

        assert!(condition.should_retry(&RetryableError::http(429, "")));
        assert!(condition.should_retry(&RetryableError::http(500, "")));
        assert!(!condition.should_retry(&RetryableError::http(400, "")));
    }

    #[test]
    fn test_api_config() {
        let config = RetryConfig::for_api();
        assert_eq!(config.max_retries, 3);
        assert!(config.retry_on.on_status_codes.contains(&429));
        assert!(config.retry_on.on_status_codes.contains(&500));
    }
}
