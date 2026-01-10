//! Filtered toolset implementation.
//!
//! This module provides `FilteredToolset`, which wraps a toolset and
//! filters tools based on a predicate function.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolDefinition, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{AbstractToolset, ToolsetTool};

/// Filters tools from a toolset based on a predicate.
///
/// Only tools where the filter function returns `true` will be available.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{FilteredToolset, FunctionToolset};
///
/// let toolset = FunctionToolset::new()
///     .tool(tool_a)
///     .tool(dangerous_tool);
///
/// // Only allow non-dangerous tools
/// let filtered = FilteredToolset::new(toolset, |_ctx, def| {
///     !def.name.contains("dangerous")
/// });
/// ```
pub struct FilteredToolset<T, F, Deps = ()> {
    inner: T,
    filter: F,
    id: Option<String>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<T, F, Deps> FilteredToolset<T, F, Deps>
where
    T: AbstractToolset<Deps>,
    F: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
{
    /// Create a new filtered toolset.
    pub fn new(inner: T, filter: F) -> Self {
        Self {
            inner,
            filter,
            id: None,
            _phantom: PhantomData,
        }
    }

    /// Set an ID for this filtered toolset.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Get the inner toolset.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

#[async_trait]
impl<T, F, Deps> AbstractToolset<Deps> for FilteredToolset<T, F, Deps>
where
    T: AbstractToolset<Deps>,
    F: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
    Deps: Send + Sync,
{
    fn id(&self) -> Option<&str> {
        self.id.as_deref().or_else(|| self.inner.id())
    }

    fn type_name(&self) -> &'static str {
        "FilteredToolset"
    }

    fn label(&self) -> String {
        format!("FilteredToolset({})", self.inner.label())
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let all_tools = self.inner.get_tools(ctx).await?;

        Ok(all_tools
            .into_iter()
            .filter(|(_, tool)| (self.filter)(ctx, &tool.tool_def))
            .collect())
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        // Verify the tool passes the filter
        if !(self.filter)(ctx, &tool.tool_def) {
            return Err(ToolError::not_found(format!(
                "Tool '{}' is not available (filtered out)",
                name
            )));
        }

        self.inner.call_tool(name, args, ctx, tool).await
    }

    async fn enter(&self) -> Result<(), ToolError> {
        self.inner.enter().await
    }

    async fn exit(&self) -> Result<(), ToolError> {
        self.inner.exit().await
    }
}

impl<T: std::fmt::Debug, F, Deps> std::fmt::Debug for FilteredToolset<T, F, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredToolset")
            .field("inner", &self.inner)
            .field("id", &self.id)
            .finish()
    }
}

/// Common filter predicates.
pub mod filters {
    use serdes_ai_tools::{RunContext, ToolDefinition};

    /// Filter that allows only tools with names in the given list.
    pub fn allow_names<Deps>(
        names: Vec<String>,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync {
        move |_, def| names.iter().any(|n| n == &def.name)
    }

    /// Filter that excludes tools with names in the given list.
    pub fn deny_names<Deps>(
        names: Vec<String>,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync {
        move |_, def| !names.iter().any(|n| n == &def.name)
    }

    /// Filter that allows tools matching a prefix.
    pub fn name_prefix<Deps>(
        prefix: String,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync {
        move |_, def| def.name.starts_with(&prefix)
    }

    /// Filter that allows tools matching a suffix.
    pub fn name_suffix<Deps>(
        suffix: String,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync {
        move |_, def| def.name.ends_with(&suffix)
    }

    /// Filter that allows tools containing a substring.
    pub fn name_contains<Deps>(
        substring: String,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync {
        move |_, def| def.name.contains(&substring)
    }

    /// Combine two filters with AND.
    pub fn and<F1, F2, Deps>(
        f1: F1,
        f2: F2,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync
    where
        F1: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
        F2: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
    {
        move |ctx, def| f1(ctx, def) && f2(ctx, def)
    }

    /// Combine two filters with OR.
    pub fn or<F1, F2, Deps>(
        f1: F1,
        f2: F2,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync
    where
        F1: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
        F2: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
    {
        move |ctx, def| f1(ctx, def) || f2(ctx, def)
    }

    /// Negate a filter.
    pub fn not<F, Deps>(f: F) -> impl Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync
    where
        F: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
    {
        move |ctx, def| !f(ctx, def)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionToolset;
    use async_trait::async_trait;
    use serdes_ai_tools::Tool;

    struct ToolA;

    #[async_trait]
    impl Tool<()> for ToolA {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("tool_a", "Tool A")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("A"))
        }
    }

    struct ToolB;

    #[async_trait]
    impl Tool<()> for ToolB {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("tool_b", "Tool B")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("B"))
        }
    }

    struct DangerousTool;

    #[async_trait]
    impl Tool<()> for DangerousTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("dangerous_delete", "Dangerous delete operation")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("Deleted!"))
        }
    }

    #[tokio::test]
    async fn test_filtered_toolset() {
        let toolset = FunctionToolset::new()
            .tool(ToolA)
            .tool(ToolB)
            .tool(DangerousTool);

        // Filter out dangerous tools
        let filtered = FilteredToolset::new(toolset, |_, def| !def.name.contains("dangerous"));

        let ctx = RunContext::minimal("test");
        let tools = filtered.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 2);
        assert!(tools.contains_key("tool_a"));
        assert!(tools.contains_key("tool_b"));
        assert!(!tools.contains_key("dangerous_delete"));
    }

    #[tokio::test]
    async fn test_filtered_toolset_call_blocked() {
        let toolset = FunctionToolset::new().tool(ToolA).tool(DangerousTool);

        let filtered = FilteredToolset::new(toolset, |_, def| !def.name.contains("dangerous"));

        let ctx = RunContext::minimal("test");

        // Create a fake tool definition for the dangerous tool
        let fake_tool = ToolsetTool::new(ToolDefinition::new("dangerous_delete", "Dangerous"));

        let result = filtered
            .call_tool("dangerous_delete", serde_json::json!({}), &ctx, &fake_tool)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_filter_predicates_allow_names() {
        let toolset = FunctionToolset::new().tool(ToolA).tool(ToolB);

        let filtered =
            FilteredToolset::new(toolset, filters::allow_names(vec!["tool_a".to_string()]));

        let ctx = RunContext::minimal("test");
        let tools = filtered.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("tool_a"));
    }

    #[tokio::test]
    async fn test_filter_predicates_deny_names() {
        let toolset = FunctionToolset::new().tool(ToolA).tool(ToolB);

        let filtered =
            FilteredToolset::new(toolset, filters::deny_names(vec!["tool_b".to_string()]));

        let ctx = RunContext::minimal("test");
        let tools = filtered.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("tool_a"));
    }

    #[tokio::test]
    async fn test_filter_predicates_combined() {
        let toolset = FunctionToolset::new()
            .tool(ToolA)
            .tool(ToolB)
            .tool(DangerousTool);

        // Allow tools starting with "tool" AND not containing "dangerous"
        let filtered = FilteredToolset::new(
            toolset,
            filters::and(
                filters::name_prefix("tool".to_string()),
                filters::not(filters::name_contains("dangerous".to_string())),
            ),
        );

        let ctx = RunContext::minimal("test");
        let tools = filtered.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 2);
    }
}
