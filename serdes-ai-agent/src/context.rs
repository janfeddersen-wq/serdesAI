//! Run context and state management.
//!
//! The context contains all information about the current agent run,
//! including dependencies, settings, and execution state.

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use serdes_ai_core::ModelSettings;
use std::sync::Arc;

/// Context for an agent run.
///
/// The context is passed to tools, instruction functions, and validators.
/// It provides access to dependencies and run metadata.
#[derive(Debug)]
pub struct RunContext<Deps> {
    /// Shared dependencies.
    pub deps: Arc<Deps>,
    /// Unique run identifier.
    pub run_id: String,
    /// Run start time.
    pub start_time: DateTime<Utc>,
    /// Model being used.
    pub model_name: String,
    /// Model settings for this run.
    pub model_settings: ModelSettings,
    /// Current tool being executed (if any).
    pub tool_name: Option<String>,
    /// Current tool call ID (if any).
    pub tool_call_id: Option<String>,
    /// Current retry count.
    pub retry_count: u32,
    /// Custom metadata.
    pub metadata: Option<JsonValue>,
}

impl<Deps> RunContext<Deps> {
    /// Create a new run context.
    pub fn new(deps: Deps, model_name: impl Into<String>) -> Self {
        Self {
            deps: Arc::new(deps),
            run_id: generate_run_id(),
            start_time: Utc::now(),
            model_name: model_name.into(),
            model_settings: ModelSettings::default(),
            tool_name: None,
            tool_call_id: None,
            retry_count: 0,
            metadata: None,
        }
    }

    /// Create with shared dependencies.
    pub fn with_shared_deps(deps: Arc<Deps>, model_name: impl Into<String>) -> Self {
        Self {
            deps,
            run_id: generate_run_id(),
            start_time: Utc::now(),
            model_name: model_name.into(),
            model_settings: ModelSettings::default(),
            tool_name: None,
            tool_call_id: None,
            retry_count: 0,
            metadata: None,
        }
    }

    /// Get a reference to the dependencies.
    pub fn deps(&self) -> &Deps {
        &self.deps
    }

    /// Get elapsed time since run started.
    pub fn elapsed(&self) -> chrono::Duration {
        Utc::now() - self.start_time
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_seconds(&self) -> i64 {
        self.elapsed().num_seconds()
    }

    /// Check if this is a retry.
    pub fn is_retry(&self) -> bool {
        self.retry_count > 0
    }

    /// Check if we're currently in a tool execution.
    pub fn in_tool(&self) -> bool {
        self.tool_name.is_some()
    }

    /// Set metadata value.
    pub fn set_metadata(&mut self, key: &str, value: impl serde::Serialize) {
        let meta = self
            .metadata
            .get_or_insert_with(|| JsonValue::Object(Default::default()));
        if let JsonValue::Object(ref mut map) = meta {
            if let Ok(v) = serde_json::to_value(value) {
                map.insert(key.to_string(), v);
            }
        }
    }

    /// Get metadata value.
    pub fn get_metadata<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.metadata
            .as_ref()
            .and_then(|m| m.get(key))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Clone with a new tool context.
    pub fn for_tool(&self, tool_name: impl Into<String>, tool_call_id: Option<String>) -> Self {
        Self {
            deps: self.deps.clone(),
            run_id: self.run_id.clone(),
            start_time: self.start_time,
            model_name: self.model_name.clone(),
            model_settings: self.model_settings.clone(),
            tool_name: Some(tool_name.into()),
            tool_call_id,
            retry_count: 0,
            metadata: self.metadata.clone(),
        }
    }

    /// Clone for a retry.
    pub fn for_retry(&self) -> Self {
        Self {
            deps: self.deps.clone(),
            run_id: self.run_id.clone(),
            start_time: self.start_time,
            model_name: self.model_name.clone(),
            model_settings: self.model_settings.clone(),
            tool_name: self.tool_name.clone(),
            tool_call_id: self.tool_call_id.clone(),
            retry_count: self.retry_count + 1,
            metadata: self.metadata.clone(),
        }
    }
}

impl<Deps: Default> Default for RunContext<Deps> {
    fn default() -> Self {
        Self::new(Deps::default(), "unknown")
    }
}

impl<Deps> Clone for RunContext<Deps> {
    fn clone(&self) -> Self {
        Self {
            deps: self.deps.clone(),
            run_id: self.run_id.clone(),
            start_time: self.start_time,
            model_name: self.model_name.clone(),
            model_settings: self.model_settings.clone(),
            tool_name: self.tool_name.clone(),
            tool_call_id: self.tool_call_id.clone(),
            retry_count: self.retry_count,
            metadata: self.metadata.clone(),
        }
    }
}

/// Generate a unique run ID.
pub fn generate_run_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Usage tracking for a run.
#[derive(Debug, Clone, Default)]
pub struct RunUsage {
    /// Total request tokens.
    pub request_tokens: u64,
    /// Total response tokens.
    pub response_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Number of model requests.
    pub request_count: u32,
    /// Number of tool calls.
    pub tool_call_count: u32,
    /// Cache creation tokens.
    pub cache_creation_tokens: Option<u64>,
    /// Cache read tokens.
    pub cache_read_tokens: Option<u64>,
}

impl RunUsage {
    /// Create new empty usage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add usage from a model request.
    pub fn add_request(&mut self, usage: serdes_ai_core::RequestUsage) {
        if let Some(req) = usage.request_tokens {
            self.request_tokens += req;
        }
        if let Some(resp) = usage.response_tokens {
            self.response_tokens += resp;
        }
        if let Some(total) = usage.total_tokens {
            self.total_tokens += total;
        } else {
            self.total_tokens = self.request_tokens + self.response_tokens;
        }
        if let Some(cache) = usage.cache_creation_tokens {
            *self.cache_creation_tokens.get_or_insert(0) += cache;
        }
        if let Some(cache) = usage.cache_read_tokens {
            *self.cache_read_tokens.get_or_insert(0) += cache;
        }
        self.request_count += 1;
    }

    /// Record a tool call.
    pub fn record_tool_call(&mut self) {
        self.tool_call_count += 1;
    }
}

/// Usage limits for a run.
#[derive(Debug, Clone, Default)]
pub struct UsageLimits {
    /// Maximum request tokens.
    pub max_request_tokens: Option<u64>,
    /// Maximum response tokens.
    pub max_response_tokens: Option<u64>,
    /// Maximum total tokens.
    pub max_total_tokens: Option<u64>,
    /// Maximum number of requests.
    pub max_requests: Option<u32>,
    /// Maximum number of tool calls.
    pub max_tool_calls: Option<u32>,
    /// Maximum run time in seconds.
    pub max_time_seconds: Option<u64>,
}

impl UsageLimits {
    /// Create new empty limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max request tokens.
    pub fn request_tokens(mut self, limit: u64) -> Self {
        self.max_request_tokens = Some(limit);
        self
    }

    /// Set max response tokens.
    pub fn response_tokens(mut self, limit: u64) -> Self {
        self.max_response_tokens = Some(limit);
        self
    }

    /// Set max total tokens.
    pub fn total_tokens(mut self, limit: u64) -> Self {
        self.max_total_tokens = Some(limit);
        self
    }

    /// Set max requests.
    pub fn requests(mut self, limit: u32) -> Self {
        self.max_requests = Some(limit);
        self
    }

    /// Set max tool calls.
    pub fn tool_calls(mut self, limit: u32) -> Self {
        self.max_tool_calls = Some(limit);
        self
    }

    /// Set max time in seconds.
    pub fn time_seconds(mut self, limit: u64) -> Self {
        self.max_time_seconds = Some(limit);
        self
    }

    /// Check usage against limits.
    pub fn check(&self, usage: &RunUsage) -> Result<(), crate::errors::UsageLimitError> {
        use crate::errors::UsageLimitError;

        if let Some(limit) = self.max_request_tokens {
            if usage.request_tokens > limit {
                return Err(UsageLimitError::RequestTokens {
                    used: usage.request_tokens,
                    limit,
                });
            }
        }

        if let Some(limit) = self.max_response_tokens {
            if usage.response_tokens > limit {
                return Err(UsageLimitError::ResponseTokens {
                    used: usage.response_tokens,
                    limit,
                });
            }
        }

        if let Some(limit) = self.max_total_tokens {
            if usage.total_tokens > limit {
                return Err(UsageLimitError::TotalTokens {
                    used: usage.total_tokens,
                    limit,
                });
            }
        }

        if let Some(limit) = self.max_requests {
            if usage.request_count > limit {
                return Err(UsageLimitError::RequestCount {
                    count: usage.request_count,
                    limit,
                });
            }
        }

        if let Some(limit) = self.max_tool_calls {
            if usage.tool_call_count > limit {
                return Err(UsageLimitError::ToolCalls {
                    count: usage.tool_call_count,
                    limit,
                });
            }
        }

        Ok(())
    }

    /// Check time limit.
    pub fn check_time(&self, elapsed_seconds: u64) -> Result<(), crate::errors::UsageLimitError> {
        if let Some(limit) = self.max_time_seconds {
            if elapsed_seconds > limit {
                return Err(crate::errors::UsageLimitError::TimeLimit {
                    elapsed_seconds,
                    limit_seconds: limit,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_context_new() {
        let ctx = RunContext::new((), "gpt-4o");
        assert_eq!(ctx.model_name, "gpt-4o");
        assert!(!ctx.run_id.is_empty());
    }

    #[test]
    fn test_run_context_metadata() {
        let mut ctx = RunContext::new((), "gpt-4o");
        ctx.set_metadata("user_id", "12345");

        let user_id: Option<String> = ctx.get_metadata("user_id");
        assert_eq!(user_id, Some("12345".to_string()));
    }

    #[test]
    fn test_run_context_for_tool() {
        let ctx = RunContext::new((), "gpt-4o");
        let tool_ctx = ctx.for_tool("search", Some("call-123".to_string()));

        assert_eq!(tool_ctx.tool_name, Some("search".to_string()));
        assert_eq!(tool_ctx.tool_call_id, Some("call-123".to_string()));
        assert!(tool_ctx.in_tool());
    }

    #[test]
    fn test_run_usage() {
        let mut usage = RunUsage::new();
        usage.add_request(serdes_ai_core::RequestUsage {
            request_tokens: Some(100),
            response_tokens: Some(50),
            total_tokens: Some(150),
            cache_creation_tokens: None,
            cache_read_tokens: None,
            details: None,
        });

        assert_eq!(usage.request_tokens, 100);
        assert_eq!(usage.response_tokens, 50);
        assert_eq!(usage.request_count, 1);
    }

    #[test]
    fn test_usage_limits() {
        let limits = UsageLimits::new().total_tokens(1000).requests(10);

        let mut usage = RunUsage::new();
        usage.total_tokens = 500;
        usage.request_count = 5;

        assert!(limits.check(&usage).is_ok());

        usage.total_tokens = 1500;
        assert!(limits.check(&usage).is_err());
    }
}
