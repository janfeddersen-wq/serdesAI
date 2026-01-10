//! Renamed toolset implementation.
//!
//! This module provides `RenamedToolset`, which renames specific tools
//! from the wrapped toolset.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{AbstractToolset, ToolsetTool};

/// Renames specific tools in a toolset.
///
/// This allows renaming individual tools without affecting others.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{RenamedToolset, FunctionToolset};
/// use std::collections::HashMap;
///
/// let toolset = FunctionToolset::new().tool(search_tool);
///
/// let renamed = RenamedToolset::new(toolset)
///     .rename("search", "find_items");
/// ```
pub struct RenamedToolset<T, Deps = ()> {
    inner: T,
    /// Maps new_name -> old_name
    name_map: HashMap<String, String>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<T, Deps> RenamedToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
{
    /// Create a new renamed toolset.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            name_map: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with a name map.
    pub fn with_map(inner: T, name_map: HashMap<String, String>) -> Self {
        Self {
            inner,
            name_map,
            _phantom: PhantomData,
        }
    }

    /// Rename a tool.
    ///
    /// # Arguments
    /// - `from`: The original tool name
    /// - `to`: The new tool name
    #[must_use]
    pub fn rename(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        // name_map is new_name -> old_name
        self.name_map.insert(to.into(), from.into());
        self
    }

    /// Get the inner toolset.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get the name map.
    #[must_use]
    pub fn name_map(&self) -> &HashMap<String, String> {
        &self.name_map
    }

    /// Get the original name for a (possibly renamed) tool.
    fn original_name<'a>(&'a self, new_name: &'a str) -> &'a str {
        self.name_map
            .get(new_name)
            .map(|s| s.as_str())
            .unwrap_or(new_name)
    }

    /// Get the new name for an original tool name.
    fn new_name(&self, original: &str) -> String {
        // Reverse lookup in name_map
        for (new, old) in &self.name_map {
            if old == original {
                return new.clone();
            }
        }
        original.to_string()
    }
}

#[async_trait]
impl<T, Deps> AbstractToolset<Deps> for RenamedToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
    Deps: Send + Sync,
{
    fn id(&self) -> Option<&str> {
        self.inner.id()
    }

    fn type_name(&self) -> &'static str {
        "RenamedToolset"
    }

    fn label(&self) -> String {
        format!("RenamedToolset({})", self.inner.label())
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let inner_tools = self.inner.get_tools(ctx).await?;

        Ok(inner_tools
            .into_iter()
            .map(|(original_name, mut tool)| {
                let new_name = self.new_name(&original_name);
                tool.tool_def.name = new_name.clone();
                (new_name, tool)
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
        let original_name = self.original_name(name);

        // Create a tool with the original name for the inner toolset
        let mut original_tool = tool.clone();
        original_tool.tool_def.name = original_name.to_string();

        self.inner
            .call_tool(original_name, args, ctx, &original_tool)
            .await
    }

    async fn enter(&self) -> Result<(), ToolError> {
        self.inner.enter().await
    }

    async fn exit(&self) -> Result<(), ToolError> {
        self.inner.exit().await
    }
}

impl<T: std::fmt::Debug, Deps> std::fmt::Debug for RenamedToolset<T, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenamedToolset")
            .field("inner", &self.inner)
            .field("name_map", &self.name_map)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionToolset;
    use async_trait::async_trait;
    use serdes_ai_tools::{Tool, ToolDefinition};

    struct SearchTool;

    #[async_trait]
    impl Tool<()> for SearchTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("search", "Search for items")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("search result"))
        }
    }

    struct QueryTool;

    #[async_trait]
    impl Tool<()> for QueryTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("query", "Query the database")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("query result"))
        }
    }

    #[test]
    fn test_original_name() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let renamed = RenamedToolset::new(toolset).rename("search", "find");

        assert_eq!(renamed.original_name("find"), "search");
        assert_eq!(renamed.original_name("other"), "other");
    }

    #[test]
    fn test_new_name() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let renamed = RenamedToolset::new(toolset).rename("search", "find");

        assert_eq!(renamed.new_name("search"), "find");
        assert_eq!(renamed.new_name("other"), "other");
    }

    #[tokio::test]
    async fn test_renamed_toolset_get_tools() {
        let toolset = FunctionToolset::new().tool(SearchTool).tool(QueryTool);
        let renamed = RenamedToolset::new(toolset).rename("search", "find_items");

        let ctx = RunContext::minimal("test");
        let tools = renamed.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 2);
        assert!(tools.contains_key("find_items"));
        assert!(tools.contains_key("query"));
        assert!(!tools.contains_key("search"));
    }

    #[tokio::test]
    async fn test_renamed_toolset_call_tool() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let renamed = RenamedToolset::new(toolset).rename("search", "find_items");

        let ctx = RunContext::minimal("test");
        let tools = renamed.get_tools(&ctx).await.unwrap();
        let tool = tools.get("find_items").unwrap();

        let result = renamed
            .call_tool("find_items", serde_json::json!({}), &ctx, tool)
            .await
            .unwrap();

        assert_eq!(result.as_text(), Some("search result"));
    }

    #[tokio::test]
    async fn test_renamed_toolset_multiple_renames() {
        let toolset = FunctionToolset::new().tool(SearchTool).tool(QueryTool);
        let renamed = RenamedToolset::new(toolset)
            .rename("search", "find")
            .rename("query", "lookup");

        let ctx = RunContext::minimal("test");
        let tools = renamed.get_tools(&ctx).await.unwrap();

        assert!(tools.contains_key("find"));
        assert!(tools.contains_key("lookup"));
    }
}
