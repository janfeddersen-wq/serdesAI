//! Approval-required toolset implementation.
//!
//! This module provides `ApprovalRequiredToolset`, which requires approval
//! before executing any tool.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolDefinition, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::{AbstractToolset, ToolsetTool};

/// Type alias for approval checker functions.
pub type ApprovalChecker<Deps> =
    dyn Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync;

/// Requires approval for tool calls.
///
/// When a tool is called, if approval is required, the toolset returns
/// `ToolError::ApprovalRequired` instead of executing the tool.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{ApprovalRequiredToolset, FunctionToolset};
///
/// let toolset = FunctionToolset::new().tool(dangerous_tool);
///
/// // Require approval for all tools
/// let approved = ApprovalRequiredToolset::new(toolset);
///
/// // Or with a custom checker
/// let approved = ApprovalRequiredToolset::with_checker(toolset, |ctx, def, args| {
///     def.name.contains("delete") || def.name.contains("modify")
/// });
/// ```
pub struct ApprovalRequiredToolset<T, Deps = ()> {
    inner: T,
    approval_checker: Arc<ApprovalChecker<Deps>>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<T, Deps> ApprovalRequiredToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
{
    /// Create a toolset that requires approval for ALL tool calls.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            approval_checker: Arc::new(|_, _, _| true), // Always require approval
            _phantom: PhantomData,
        }
    }

    /// Create a toolset with a custom approval checker.
    ///
    /// The checker returns `true` if approval is required for the given
    /// tool call, `false` if the call can proceed without approval.
    pub fn with_checker<F>(inner: T, checker: F) -> Self
    where
        F: Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync + 'static,
    {
        Self {
            inner,
            approval_checker: Arc::new(checker),
            _phantom: PhantomData,
        }
    }

    /// Get the inner toolset.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

#[async_trait]
impl<T, Deps> AbstractToolset<Deps> for ApprovalRequiredToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
    Deps: Send + Sync,
{
    fn id(&self) -> Option<&str> {
        self.inner.id()
    }

    fn type_name(&self) -> &'static str {
        "ApprovalRequiredToolset"
    }

    fn label(&self) -> String {
        format!("ApprovalRequiredToolset({})", self.inner.label())
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        self.inner.get_tools(ctx).await
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        // Check if approval is required
        if (self.approval_checker)(ctx, &tool.tool_def, &args) {
            return Err(ToolError::ApprovalRequired {
                tool_name: name.to_string(),
                args,
            });
        }

        // No approval needed, proceed with the call
        self.inner.call_tool(name, args, ctx, tool).await
    }

    async fn enter(&self) -> Result<(), ToolError> {
        self.inner.enter().await
    }

    async fn exit(&self) -> Result<(), ToolError> {
        self.inner.exit().await
    }
}

impl<T: std::fmt::Debug, Deps> std::fmt::Debug for ApprovalRequiredToolset<T, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApprovalRequiredToolset")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Common approval checkers.
pub mod checkers {
    use serde_json::Value as JsonValue;
    use serdes_ai_tools::{RunContext, ToolDefinition};

    /// Always require approval.
    pub fn always<Deps>(
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync {
        |_, _, _| true
    }

    /// Never require approval.
    pub fn never<Deps>(
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync {
        |_, _, _| false
    }

    /// Require approval for tools with names containing any of the given substrings.
    pub fn name_contains<Deps>(
        substrings: Vec<String>,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync {
        move |_, def, _| substrings.iter().any(|s| def.name.contains(s.as_str()))
    }

    /// Require approval for tools with names in the given list.
    pub fn tool_names<Deps>(
        names: Vec<String>,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync {
        move |_, def, _| names.iter().any(|n| n == &def.name)
    }

    /// Require approval for tools with names matching a prefix.
    pub fn name_prefix<Deps>(
        prefix: String,
    ) -> impl Fn(&RunContext<Deps>, &ToolDefinition, &JsonValue) -> bool + Send + Sync {
        move |_, def, _| def.name.starts_with(&prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionToolset;
    use async_trait::async_trait;
    use serdes_ai_tools::Tool;

    struct SafeTool;

    #[async_trait]
    impl Tool<()> for SafeTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("safe_read", "Safe read operation")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("read data"))
        }
    }

    struct DangerousTool;

    #[async_trait]
    impl Tool<()> for DangerousTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("delete_all", "Delete all data")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("deleted"))
        }
    }

    #[tokio::test]
    async fn test_approval_required_all() {
        let toolset = FunctionToolset::new().tool(SafeTool);
        let approved = ApprovalRequiredToolset::new(toolset);

        let ctx = RunContext::minimal("test");
        let tools = approved.get_tools(&ctx).await.unwrap();
        let tool = tools.get("safe_read").unwrap();

        let result = approved
            .call_tool("safe_read", serde_json::json!({}), &ctx, tool)
            .await;

        assert!(matches!(result, Err(ToolError::ApprovalRequired { .. })));
    }

    #[tokio::test]
    async fn test_approval_required_selective() {
        let toolset = FunctionToolset::new().tool(SafeTool).tool(DangerousTool);

        // Only require approval for delete operations
        let approved =
            ApprovalRequiredToolset::with_checker(toolset, |_, def, _| def.name.contains("delete"));

        let ctx = RunContext::minimal("test");
        let tools = approved.get_tools(&ctx).await.unwrap();

        // Safe tool should work
        let safe_tool = tools.get("safe_read").unwrap();
        let result = approved
            .call_tool("safe_read", serde_json::json!({}), &ctx, safe_tool)
            .await;
        assert!(result.is_ok());

        // Dangerous tool should require approval
        let dangerous_tool = tools.get("delete_all").unwrap();
        let result = approved
            .call_tool("delete_all", serde_json::json!({}), &ctx, dangerous_tool)
            .await;
        assert!(matches!(result, Err(ToolError::ApprovalRequired { .. })));
    }

    #[tokio::test]
    async fn test_approval_never() {
        let toolset = FunctionToolset::new().tool(SafeTool);
        let approved = ApprovalRequiredToolset::with_checker(toolset, checkers::never());

        let ctx = RunContext::minimal("test");
        let tools = approved.get_tools(&ctx).await.unwrap();
        let tool = tools.get("safe_read").unwrap();

        let result = approved
            .call_tool("safe_read", serde_json::json!({}), &ctx, tool)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_approval_checker_name_contains() {
        let toolset = FunctionToolset::new().tool(SafeTool).tool(DangerousTool);
        let approved = ApprovalRequiredToolset::with_checker(
            toolset,
            checkers::name_contains(vec!["delete".to_string(), "remove".to_string()]),
        );

        let ctx = RunContext::minimal("test");
        let tools = approved.get_tools(&ctx).await.unwrap();

        let dangerous = tools.get("delete_all").unwrap();
        let result = approved
            .call_tool("delete_all", serde_json::json!({}), &ctx, dangerous)
            .await;

        assert!(matches!(result, Err(ToolError::ApprovalRequired { .. })));
    }
}
