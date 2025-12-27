//! Prepared toolset implementation.
//!
//! This module provides `PreparedToolset`, which modifies tool definitions
//! at runtime using a prepare function.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolDefinition, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{AbstractToolset, ToolsetTool};

/// Prepares/modifies tool definitions at runtime.
///
/// This allows dynamically modifying tool definitions based on the
/// current context, such as adding dynamic descriptions or hiding
/// tools based on user permissions.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{PreparedToolset, FunctionToolset};
///
/// let toolset = FunctionToolset::new().tool(admin_tool);
///
/// // Hide admin tools for non-admin users
/// let prepared = PreparedToolset::new(toolset, |ctx, defs| {
///     if ctx.deps.is_admin {
///         Some(defs)
///     } else {
///         Some(defs.into_iter().filter(|d| !d.name.starts_with("admin_")).collect())
///     }
/// });
/// ```
pub struct PreparedToolset<T, F, Deps = ()> {
    inner: T,
    prepare_fn: F,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<T, F, Deps> PreparedToolset<T, F, Deps>
where
    T: AbstractToolset<Deps>,
    F: Fn(&RunContext<Deps>, Vec<ToolDefinition>) -> Option<Vec<ToolDefinition>> + Send + Sync,
{
    /// Create a new prepared toolset.
    pub fn new(inner: T, prepare_fn: F) -> Self {
        Self {
            inner,
            prepare_fn,
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
impl<T, F, Deps> AbstractToolset<Deps> for PreparedToolset<T, F, Deps>
where
    T: AbstractToolset<Deps>,
    F: Fn(&RunContext<Deps>, Vec<ToolDefinition>) -> Option<Vec<ToolDefinition>> + Send + Sync,
    Deps: Send + Sync,
{
    fn id(&self) -> Option<&str> {
        self.inner.id()
    }

    fn type_name(&self) -> &'static str {
        "PreparedToolset"
    }

    fn label(&self) -> String {
        format!("PreparedToolset({})", self.inner.label())
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let inner_tools = self.inner.get_tools(ctx).await?;

        // Extract definitions for the prepare function
        let defs: Vec<ToolDefinition> = inner_tools
            .values()
            .map(|t| t.tool_def.clone())
            .collect();

        // Apply the prepare function
        let prepared_defs = match (self.prepare_fn)(ctx, defs) {
            Some(defs) => defs,
            None => return Ok(HashMap::new()), // Return empty if prepare returns None
        };

        // Build result, keeping only tools that are in the prepared definitions
        let prepared_names: std::collections::HashSet<_> =
            prepared_defs.iter().map(|d| d.name.clone()).collect();

        // Create a map of prepared definitions
        let def_map: HashMap<String, ToolDefinition> = prepared_defs
            .into_iter()
            .map(|d| (d.name.clone(), d))
            .collect();

        Ok(inner_tools
            .into_iter()
            .filter(|(name, _)| prepared_names.contains(name))
            .map(|(name, mut tool)| {
                // Update with the potentially modified definition
                if let Some(prepared_def) = def_map.get(&name) {
                    tool.tool_def = prepared_def.clone();
                }
                (name, tool)
            })
            .collect())
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        self.inner.call_tool(name, args, ctx, tool).await
    }

    async fn enter(&self) -> Result<(), ToolError> {
        self.inner.enter().await
    }

    async fn exit(&self) -> Result<(), ToolError> {
        self.inner.exit().await
    }
}

impl<T: std::fmt::Debug, F, Deps> std::fmt::Debug for PreparedToolset<T, F, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedToolset")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Common prepare functions.
pub mod preparers {
    use serdes_ai_tools::{RunContext, ToolDefinition};

    /// Add a suffix to all tool descriptions.
    pub fn add_description_suffix<Deps>(
        suffix: &str,
    ) -> impl Fn(&RunContext<Deps>, Vec<ToolDefinition>) -> Option<Vec<ToolDefinition>> + Send + Sync + '_
    {
        move |_, defs| {
            Some(
                defs.into_iter()
                    .map(|mut d| {
                        d.description = format!("{} {}", d.description, suffix);
                        d
                    })
                    .collect(),
            )
        }
    }

    /// Filter tools based on a predicate.
    pub fn filter<Deps, F>(
        pred: F,
    ) -> impl Fn(&RunContext<Deps>, Vec<ToolDefinition>) -> Option<Vec<ToolDefinition>> + Send + Sync
    where
        F: Fn(&RunContext<Deps>, &ToolDefinition) -> bool + Send + Sync,
    {
        move |ctx, defs| Some(defs.into_iter().filter(|d| pred(ctx, d)).collect())
    }

    /// Sort tools by name.
    pub fn sort_by_name<Deps>(
    ) -> impl Fn(&RunContext<Deps>, Vec<ToolDefinition>) -> Option<Vec<ToolDefinition>> + Send + Sync
    {
        |_, mut defs| {
            defs.sort_by(|a, b| a.name.cmp(&b.name));
            Some(defs)
        }
    }

    /// Limit the number of tools.
    pub fn limit<Deps>(
        max: usize,
    ) -> impl Fn(&RunContext<Deps>, Vec<ToolDefinition>) -> Option<Vec<ToolDefinition>> + Send + Sync
    {
        move |_, defs| Some(defs.into_iter().take(max).collect())
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

    struct AdminTool;

    #[async_trait]
    impl Tool<()> for AdminTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("admin_delete", "Admin delete")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("Deleted"))
        }
    }

    #[tokio::test]
    async fn test_prepared_toolset_filter() {
        let toolset = FunctionToolset::new().tool(ToolA).tool(AdminTool);

        // Hide admin tools
        let prepared = PreparedToolset::new(toolset, |_, defs| {
            Some(
                defs.into_iter()
                    .filter(|d| !d.name.starts_with("admin_"))
                    .collect(),
            )
        });

        let ctx = RunContext::minimal("test");
        let tools = prepared.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("tool_a"));
        assert!(!tools.contains_key("admin_delete"));
    }

    #[tokio::test]
    async fn test_prepared_toolset_modify_description() {
        let toolset = FunctionToolset::new().tool(ToolA);

        let prepared = PreparedToolset::new(toolset, |_, defs| {
            Some(
                defs.into_iter()
                    .map(|mut d| {
                        d.description = format!("[MODIFIED] {}", d.description);
                        d
                    })
                    .collect(),
            )
        });

        let ctx = RunContext::minimal("test");
        let tools = prepared.get_tools(&ctx).await.unwrap();

        let tool = tools.get("tool_a").unwrap();
        assert!(tool.tool_def.description.starts_with("[MODIFIED]"));
    }

    #[tokio::test]
    async fn test_prepared_toolset_returns_none() {
        let toolset = FunctionToolset::new().tool(ToolA);

        // Return None to hide all tools
        let prepared = PreparedToolset::new(toolset, |_, _| None);

        let ctx = RunContext::minimal("test");
        let tools = prepared.get_tools(&ctx).await.unwrap();

        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn test_prepared_toolset_call_still_works() {
        let toolset = FunctionToolset::new().tool(ToolA);

        let prepared = PreparedToolset::new(toolset, |_, defs| Some(defs));

        let ctx = RunContext::minimal("test");
        let tools = prepared.get_tools(&ctx).await.unwrap();
        let tool = tools.get("tool_a").unwrap();

        let result = prepared
            .call_tool("tool_a", serde_json::json!({}), &ctx, tool)
            .await
            .unwrap();

        assert_eq!(result.as_text(), Some("A"));
    }
}
