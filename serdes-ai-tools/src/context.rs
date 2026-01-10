//! Run context for tool execution.
//!
//! This module provides the `RunContext` type which carries contextual information
//! to tools during execution, including dependencies, model info, and usage tracking.

use chrono::{DateTime, Utc};
use serdes_ai_core::{identifier::generate_run_id, ModelSettings, RunUsage};
use std::sync::Arc;

/// Context passed to tools during execution.
///
/// The `RunContext` provides tools with access to:
/// - User-provided dependencies (database connections, API clients, etc.)
/// - Run metadata (ID, start time, model name)
/// - Retry information
/// - Current usage statistics
///
/// # Type Parameters
///
/// - `Deps`: The type of dependencies available to tools. Defaults to `()`.
///
/// # Example
///
/// ```rust
/// use serdes_ai_tools::RunContext;
/// use std::sync::Arc;
///
/// struct MyDeps {
///     api_key: String,
/// }
///
/// let deps = MyDeps { api_key: "secret".into() };
/// let ctx = RunContext::new(deps, "gpt-4");
///
/// // Access deps in a tool
/// assert_eq!(ctx.deps.api_key, "secret");
/// ```
#[derive(Debug, Clone)]
pub struct RunContext<Deps = ()> {
    /// User-provided dependencies.
    pub deps: Arc<Deps>,

    /// Unique identifier for this run.
    pub run_id: String,

    /// When this run started.
    pub start_time: DateTime<Utc>,

    /// Current retry count for tool call.
    pub retry_count: u32,

    /// Maximum retries allowed.
    pub max_retries: u32,

    /// Name of the tool being called (if in a tool call).
    pub tool_name: Option<String>,

    /// Tool call ID (if in a tool call).
    pub tool_call_id: Option<String>,

    /// Name of the model being used.
    pub model_name: String,

    /// Current model settings.
    pub model_settings: ModelSettings,

    /// Usage statistics so far.
    pub usage: RunUsage,

    /// Custom metadata.
    pub metadata: Option<serde_json::Value>,

    /// Whether partial output is being generated.
    pub partial_output: bool,
}

impl<Deps> RunContext<Deps> {
    /// Create a new run context.
    #[must_use]
    pub fn new(deps: Deps, model_name: impl Into<String>) -> Self {
        Self {
            deps: Arc::new(deps),
            run_id: generate_run_id(),
            start_time: Utc::now(),
            retry_count: 0,
            max_retries: 3,
            tool_name: None,
            tool_call_id: None,
            model_name: model_name.into(),
            model_settings: ModelSettings::default(),
            usage: RunUsage::default(),
            metadata: None,
            partial_output: false,
        }
    }

    /// Create a context from existing Arc'd deps.
    #[must_use]
    pub fn from_arc(deps: Arc<Deps>, model_name: impl Into<String>) -> Self {
        Self {
            deps,
            run_id: generate_run_id(),
            start_time: Utc::now(),
            retry_count: 0,
            max_retries: 3,
            tool_name: None,
            tool_call_id: None,
            model_name: model_name.into(),
            model_settings: ModelSettings::default(),
            usage: RunUsage::default(),
            metadata: None,
            partial_output: false,
        }
    }

    /// Set the run ID.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// Set max retries.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set model settings.
    #[must_use]
    pub fn with_model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self
    }

    /// Set metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the tool context for a tool call.
    #[must_use]
    pub fn with_tool_context(
        mut self,
        tool_name: impl Into<String>,
        tool_call_id: Option<String>,
    ) -> Self {
        self.tool_name = Some(tool_name.into());
        self.tool_call_id = tool_call_id;
        self
    }

    /// Set partial output mode.
    #[must_use]
    pub fn with_partial_output(mut self, partial: bool) -> Self {
        self.partial_output = partial;
        self
    }

    /// Increment the retry count.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if we can retry.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Get elapsed time since start.
    #[must_use]
    pub fn elapsed(&self) -> chrono::Duration {
        Utc::now() - self.start_time
    }

    /// Get elapsed time in seconds.
    #[must_use]
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().num_milliseconds() as f64 / 1000.0
    }

    /// Check if we're currently in a tool call.
    #[must_use]
    pub fn in_tool_call(&self) -> bool {
        self.tool_name.is_some()
    }

    /// Create a child context for a tool call.
    #[must_use]
    pub fn for_tool(&self, tool_name: impl Into<String>, tool_call_id: Option<String>) -> Self {
        Self {
            deps: Arc::clone(&self.deps),
            run_id: self.run_id.clone(),
            start_time: self.start_time,
            retry_count: 0,
            max_retries: self.max_retries,
            tool_name: Some(tool_name.into()),
            tool_call_id,
            model_name: self.model_name.clone(),
            model_settings: self.model_settings.clone(),
            usage: self.usage.clone(),
            metadata: self.metadata.clone(),
            partial_output: self.partial_output,
        }
    }

    /// Create a copy with updated usage.
    #[must_use]
    pub fn with_usage(mut self, usage: RunUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Replace dependencies (for testing).
    #[must_use]
    pub fn with_deps<NewDeps>(self, new_deps: NewDeps) -> RunContext<NewDeps> {
        RunContext {
            deps: Arc::new(new_deps),
            run_id: self.run_id,
            start_time: self.start_time,
            retry_count: self.retry_count,
            max_retries: self.max_retries,
            tool_name: self.tool_name,
            tool_call_id: self.tool_call_id,
            model_name: self.model_name,
            model_settings: self.model_settings,
            usage: self.usage,
            metadata: self.metadata,
            partial_output: self.partial_output,
        }
    }
}

impl<Deps: Default> Default for RunContext<Deps> {
    fn default() -> Self {
        Self::new(Deps::default(), "default")
    }
}

impl RunContext<()> {
    /// Create a minimal context without dependencies.
    #[must_use]
    pub fn minimal(model_name: impl Into<String>) -> Self {
        Self::new((), model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default)]
    struct TestDeps {
        value: i32,
    }

    #[test]
    fn test_run_context_new() {
        let ctx = RunContext::new(TestDeps { value: 42 }, "gpt-4");
        assert_eq!(ctx.deps.value, 42);
        assert_eq!(ctx.model_name, "gpt-4");
        assert!(ctx.run_id.starts_with("run_"));
        assert_eq!(ctx.retry_count, 0);
    }

    #[test]
    fn test_run_context_minimal() {
        let ctx = RunContext::minimal("claude-3");
        assert_eq!(ctx.model_name, "claude-3");
    }

    #[test]
    fn test_run_context_with_tool_context() {
        let ctx =
            RunContext::minimal("gpt-4").with_tool_context("my_tool", Some("call_123".to_string()));
        assert_eq!(ctx.tool_name, Some("my_tool".to_string()));
        assert_eq!(ctx.tool_call_id, Some("call_123".to_string()));
        assert!(ctx.in_tool_call());
    }

    #[test]
    fn test_increment_retry() {
        let mut ctx = RunContext::minimal("gpt-4").with_max_retries(3);
        assert!(ctx.can_retry());
        ctx.increment_retry();
        ctx.increment_retry();
        ctx.increment_retry();
        assert!(!ctx.can_retry());
    }

    #[test]
    fn test_for_tool() {
        let ctx = RunContext::new(TestDeps { value: 10 }, "gpt-4");
        let tool_ctx = ctx.for_tool("test_tool", Some("id1".to_string()));

        // Same deps
        assert_eq!(tool_ctx.deps.value, 10);
        // Same run ID
        assert_eq!(tool_ctx.run_id, ctx.run_id);
        // Tool info set
        assert_eq!(tool_ctx.tool_name, Some("test_tool".to_string()));
        // Reset retry count
        assert_eq!(tool_ctx.retry_count, 0);
    }

    #[test]
    fn test_with_deps() {
        let ctx = RunContext::minimal("gpt-4");
        let new_ctx = ctx.with_deps(TestDeps { value: 99 });
        assert_eq!(new_ctx.deps.value, 99);
    }

    #[test]
    fn test_elapsed() {
        let ctx = RunContext::minimal("gpt-4");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = ctx.elapsed_secs();
        assert!(elapsed >= 0.01);
    }

    #[test]
    fn test_default() {
        let ctx: RunContext<TestDeps> = RunContext::default();
        assert_eq!(ctx.deps.value, 0);
        assert_eq!(ctx.model_name, "default");
    }
}
