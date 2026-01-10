//! Token usage tracking for model requests.
//!
//! This module provides types for tracking token usage across requests and runs,
//! as well as usage limit checking.

use serde::{Deserialize, Serialize};

use crate::errors::{UsageLimitExceeded, UsageLimitType};

/// Token usage for a single request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestUsage {
    /// Number of tokens in the request/prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_tokens: Option<u64>,
    /// Number of tokens in the response/completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_tokens: Option<u64>,
    /// Total tokens (request + response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Tokens used to create cache entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
    /// Tokens read from cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Provider-specific usage details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl RequestUsage {
    /// Create a new empty usage record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create usage with request and response tokens.
    #[must_use]
    pub fn with_tokens(request_tokens: u64, response_tokens: u64) -> Self {
        Self {
            request_tokens: Some(request_tokens),
            response_tokens: Some(response_tokens),
            total_tokens: Some(request_tokens + response_tokens),
            ..Self::default()
        }
    }

    /// Set request tokens.
    #[must_use]
    pub fn request_tokens(mut self, tokens: u64) -> Self {
        self.request_tokens = Some(tokens);
        self.recalculate_total();
        self
    }

    /// Set response tokens.
    #[must_use]
    pub fn response_tokens(mut self, tokens: u64) -> Self {
        self.response_tokens = Some(tokens);
        self.recalculate_total();
        self
    }

    /// Set cache creation tokens.
    #[must_use]
    pub fn cache_creation_tokens(mut self, tokens: u64) -> Self {
        self.cache_creation_tokens = Some(tokens);
        self
    }

    /// Set cache read tokens.
    #[must_use]
    pub fn cache_read_tokens(mut self, tokens: u64) -> Self {
        self.cache_read_tokens = Some(tokens);
        self
    }

    /// Set details.
    #[must_use]
    pub fn details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Merge another usage record into this one.
    pub fn merge(&mut self, other: &RequestUsage) {
        self.request_tokens = match (self.request_tokens, other.request_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.response_tokens = match (self.response_tokens, other.response_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.cache_creation_tokens = match (self.cache_creation_tokens, other.cache_creation_tokens)
        {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.cache_read_tokens = match (self.cache_read_tokens, other.cache_read_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.recalculate_total();
    }

    /// Recalculate total from request and response.
    fn recalculate_total(&mut self) {
        self.total_tokens = match (self.request_tokens, self.response_tokens) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }

    /// Get total tokens, calculating if not set.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total_tokens
            .unwrap_or_else(|| self.request_tokens.unwrap_or(0) + self.response_tokens.unwrap_or(0))
    }

    /// Check if this usage record has any data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.request_tokens.is_none()
            && self.response_tokens.is_none()
            && self.total_tokens.is_none()
    }
}

impl std::ops::Add for RequestUsage {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.merge(&rhs);
        self
    }
}

impl std::ops::AddAssign for RequestUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.merge(&rhs);
    }
}

/// Accumulated usage for an entire run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunUsage {
    /// Individual request usages.
    pub requests: Vec<RequestUsage>,
    /// Total request tokens across all requests.
    pub total_request_tokens: u64,
    /// Total response tokens across all requests.
    pub total_response_tokens: u64,
    /// Total tokens across all requests.
    pub total_tokens: u64,
}

impl RunUsage {
    /// Create a new empty run usage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a request's usage.
    pub fn add_request(&mut self, usage: RequestUsage) {
        self.total_request_tokens += usage.request_tokens.unwrap_or(0);
        self.total_response_tokens += usage.response_tokens.unwrap_or(0);
        self.total_tokens += usage.total();
        self.requests.push(usage);
    }

    /// Get the number of requests.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Check if there's no usage data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Get average tokens per request.
    #[must_use]
    pub fn avg_tokens_per_request(&self) -> f64 {
        if self.requests.is_empty() {
            0.0
        } else {
            self.total_tokens as f64 / self.requests.len() as f64
        }
    }
}

/// Usage limits for a run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageLimits {
    /// Maximum request tokens per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_request_tokens: Option<u64>,
    /// Maximum response tokens per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_tokens: Option<u64>,
    /// Maximum total tokens for the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_total_tokens: Option<u64>,
    /// Maximum number of requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_requests: Option<u64>,
}

impl UsageLimits {
    /// Create new empty limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max request tokens.
    #[must_use]
    pub fn max_request_tokens(mut self, tokens: u64) -> Self {
        self.max_request_tokens = Some(tokens);
        self
    }

    /// Set max response tokens.
    #[must_use]
    pub fn max_response_tokens(mut self, tokens: u64) -> Self {
        self.max_response_tokens = Some(tokens);
        self
    }

    /// Set max total tokens.
    #[must_use]
    pub fn max_total_tokens(mut self, tokens: u64) -> Self {
        self.max_total_tokens = Some(tokens);
        self
    }

    /// Set max requests.
    #[must_use]
    pub fn max_requests(mut self, requests: u64) -> Self {
        self.max_requests = Some(requests);
        self
    }

    /// Check if usage exceeds limits.
    ///
    /// Returns `Ok(())` if within limits, or an error describing which limit was exceeded.
    pub fn check(&self, usage: &RunUsage) -> Result<(), UsageLimitExceeded> {
        if let Some(max) = self.max_request_tokens {
            if usage.total_request_tokens > max {
                return Err(UsageLimitExceeded::new(
                    UsageLimitType::RequestTokens,
                    usage.total_request_tokens,
                    max,
                ));
            }
        }

        if let Some(max) = self.max_response_tokens {
            if usage.total_response_tokens > max {
                return Err(UsageLimitExceeded::new(
                    UsageLimitType::ResponseTokens,
                    usage.total_response_tokens,
                    max,
                ));
            }
        }

        if let Some(max) = self.max_total_tokens {
            if usage.total_tokens > max {
                return Err(UsageLimitExceeded::new(
                    UsageLimitType::TotalTokens,
                    usage.total_tokens,
                    max,
                ));
            }
        }

        if let Some(max) = self.max_requests {
            let count = usage.request_count() as u64;
            if count > max {
                return Err(UsageLimitExceeded::new(
                    UsageLimitType::Requests,
                    count,
                    max,
                ));
            }
        }

        Ok(())
    }

    /// Check if any limits are set.
    #[must_use]
    pub fn has_limits(&self) -> bool {
        self.max_request_tokens.is_some()
            || self.max_response_tokens.is_some()
            || self.max_total_tokens.is_some()
            || self.max_requests.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_usage_new() {
        let usage = RequestUsage::new();
        assert!(usage.is_empty());
    }

    #[test]
    fn test_request_usage_with_tokens() {
        let usage = RequestUsage::with_tokens(100, 50);
        assert_eq!(usage.request_tokens, Some(100));
        assert_eq!(usage.response_tokens, Some(50));
        assert_eq!(usage.total_tokens, Some(150));
    }

    #[test]
    fn test_request_usage_merge() {
        let mut usage1 = RequestUsage::with_tokens(100, 50);
        let usage2 = RequestUsage::with_tokens(200, 100);
        usage1.merge(&usage2);
        assert_eq!(usage1.request_tokens, Some(300));
        assert_eq!(usage1.response_tokens, Some(150));
        assert_eq!(usage1.total(), 450);
    }

    #[test]
    fn test_run_usage() {
        let mut run = RunUsage::new();
        run.add_request(RequestUsage::with_tokens(100, 50));
        run.add_request(RequestUsage::with_tokens(200, 100));

        assert_eq!(run.request_count(), 2);
        assert_eq!(run.total_request_tokens, 300);
        assert_eq!(run.total_response_tokens, 150);
        assert_eq!(run.total_tokens, 450);
    }

    #[test]
    fn test_usage_limits_check_pass() {
        let limits = UsageLimits::new().max_total_tokens(1000).max_requests(10);

        let mut run = RunUsage::new();
        run.add_request(RequestUsage::with_tokens(100, 50));

        assert!(limits.check(&run).is_ok());
    }

    #[test]
    fn test_usage_limits_check_fail() {
        let limits = UsageLimits::new().max_total_tokens(100);

        let mut run = RunUsage::new();
        run.add_request(RequestUsage::with_tokens(100, 50));

        let result = limits.check(&run);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.limit_type, UsageLimitType::TotalTokens);
    }

    #[test]
    fn test_serde_roundtrip() {
        let usage = RequestUsage::with_tokens(100, 50)
            .cache_creation_tokens(10)
            .cache_read_tokens(5);
        let json = serde_json::to_string(&usage).unwrap();
        let parsed: RequestUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, parsed);
    }
}
