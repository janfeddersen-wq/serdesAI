//! # serdes-ai-retries
//!
//! Retry strategies and error handling for serdes-ai.
//!
//! This crate provides flexible retry mechanisms for handling transient
//! failures in LLM API calls.
//!
//! ## Core Concepts
//!
//! - **[`RetryConfig`]**: Configure retry behavior
//! - **[`WaitStrategy`]**: Define how long to wait between retries
//! - **[`RetryCondition`]**: Determine which errors are retryable
//! - **[`with_retry`]**: Execute operations with automatic retries
//! - **[`RetryClient`]**: HTTP client with built-in retry support
//!
//! ## Wait Strategies
//!
//! - [`WaitStrategy::Fixed`]: Constant delay between attempts
//! - [`WaitStrategy::ExponentialBackoff`]: Exponential delay with cap
//! - [`WaitStrategy::ExponentialJitter`]: Exponential with randomization
//! - [`WaitStrategy::Linear`]: Linearly increasing delay
//! - [`WaitStrategy::RetryAfter`]: Respect server's Retry-After header
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_retries::{with_retry, RetryConfig, RetryableError};
//! use std::time::Duration;
//!
//! let config = RetryConfig::new()
//!     .max_retries(3)
//!     .exponential_jitter(
//!         Duration::from_millis(100),
//!         Duration::from_secs(10),
//!         0.1,
//!     );
//!
//! let result = with_retry(&config, || async {
//!     // Your async operation
//!     Ok::<_, RetryableError>("success")
//! }).await?;
//! ```
//!
//! ## HTTP Client with Retries
//!
//! ```ignore
//! use serdes_ai_retries::RetryClient;
//!
//! let client = RetryClient::for_api();
//!
//! let response = client.get("https://api.example.com/data").await?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod backoff;
pub mod config;
pub mod error;
pub mod executor;
pub mod strategy;
pub mod transport;

// Re-exports
pub use backoff::{ExponentialBackoff, ExponentialBackoffBuilder, FixedDelay, LinearBackoff};
pub use config::{RetryCondition, RetryConfig, WaitStrategy};
pub use error::{RetryResult, RetryableError};
pub use executor::{with_retry, with_retry_state, AttemptInfo, Retry, RetryState};
pub use strategy::{NoRetry, RetryStrategy};
pub use transport::{RetryClient, RetryClientBuilder};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        with_retry, ExponentialBackoff, Retry, RetryClient, RetryConfig, RetryResult,
        RetryStrategy, RetryableError, WaitStrategy,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let config = RetryConfig::new().max_retries(5);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_exponential_backoff() {
        let backoff = ExponentialBackoff::new();
        assert_eq!(backoff.max_retries, 3);
    }

    #[test]
    fn test_fixed_delay() {
        let delay = FixedDelay::new(Duration::from_secs(1), 5);
        assert_eq!(delay.max_retries, 5);
    }
}
