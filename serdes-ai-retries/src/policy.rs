//! Retry policy for determining what to retry.

use crate::error::RetryableError;

/// Policy for determining which errors should be retried.
pub trait RetryPolicy: Send + Sync {
    /// Check if the error should be retried.
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> bool;
}

/// Policy that retries all retryable errors.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultPolicy {
    /// Maximum attempts.
    pub max_attempts: u32,
}

impl DefaultPolicy {
    /// Create a new default policy.
    pub fn new(max_attempts: u32) -> Self {
        Self { max_attempts }
    }
}

impl RetryPolicy for DefaultPolicy {
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> bool {
        attempt < self.max_attempts && error.is_retryable()
    }
}

/// Policy that retries only specific status codes.
#[derive(Debug, Clone)]
pub struct StatusCodePolicy {
    /// Status codes to retry.
    pub codes: Vec<u16>,
    /// Maximum attempts.
    pub max_attempts: u32,
}

impl StatusCodePolicy {
    /// Create a new status code policy.
    pub fn new(codes: Vec<u16>, max_attempts: u32) -> Self {
        Self { codes, max_attempts }
    }

    /// Create for server errors (5xx).
    pub fn server_errors(max_attempts: u32) -> Self {
        Self::new((500..=599).collect(), max_attempts)
    }

    /// Create for rate limits.
    pub fn rate_limit(max_attempts: u32) -> Self {
        Self::new(vec![429], max_attempts)
    }
}

impl RetryPolicy for StatusCodePolicy {
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> bool {
        if attempt >= self.max_attempts {
            return false;
        }

        match error.status() {
            Some(status) => self.codes.contains(&status),
            None => false,
        }
    }
}

/// Combine multiple policies with OR logic.
pub struct CombinedPolicy {
    policies: Vec<Box<dyn RetryPolicy>>,
}

impl std::fmt::Debug for CombinedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedPolicy")
            .field("policy_count", &self.policies.len())
            .finish()
    }
}

impl CombinedPolicy {
    /// Create a new combined policy.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    /// Add a policy.
    pub fn add(mut self, policy: impl RetryPolicy + 'static) -> Self {
        self.policies.push(Box::new(policy));
        self
    }
}

impl Default for CombinedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryPolicy for CombinedPolicy {
    fn should_retry(&self, error: &RetryableError, attempt: u32) -> bool {
        self.policies
            .iter()
            .any(|p| p.should_retry(error, attempt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = DefaultPolicy::new(3);

        assert!(policy.should_retry(&RetryableError::http(500, ""), 0));
        assert!(policy.should_retry(&RetryableError::http(500, ""), 2));
        assert!(!policy.should_retry(&RetryableError::http(500, ""), 3));
        assert!(!policy.should_retry(&RetryableError::http(400, ""), 0));
    }

    #[test]
    fn test_status_code_policy() {
        let policy = StatusCodePolicy::server_errors(3);

        assert!(policy.should_retry(&RetryableError::http(500, ""), 0));
        assert!(policy.should_retry(&RetryableError::http(503, ""), 0));
        assert!(!policy.should_retry(&RetryableError::http(400, ""), 0));
        assert!(!policy.should_retry(&RetryableError::http(429, ""), 0));
    }

    #[test]
    fn test_combined_policy() {
        let policy = CombinedPolicy::new()
            .add(StatusCodePolicy::server_errors(3))
            .add(StatusCodePolicy::rate_limit(3));

        assert!(policy.should_retry(&RetryableError::http(500, ""), 0));
        assert!(policy.should_retry(&RetryableError::http(429, ""), 0));
        assert!(!policy.should_retry(&RetryableError::http(400, ""), 0));
    }
}
