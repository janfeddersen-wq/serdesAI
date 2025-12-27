//! Prefixed toolset implementation.
//!
//! This module provides `PrefixedToolset`, which adds a prefix to all
//! tool names from the wrapped toolset.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{AbstractToolset, ToolsetTool};

/// Adds a prefix to all tool names.
///
/// This is useful for avoiding name conflicts when combining multiple
/// toolsets that might have tools with the same name.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{PrefixedToolset, FunctionToolset};
///
/// let tools1 = FunctionToolset::new().tool(search_tool);
/// let tools2 = FunctionToolset::new().tool(search_tool);
///
/// // Prefix to avoid conflicts
/// let prefixed1 = PrefixedToolset::new(tools1, "web");
/// let prefixed2 = PrefixedToolset::new(tools2, "local");
///
/// // Now we have "web_search" and "local_search"
/// ```
pub struct PrefixedToolset<T, Deps = ()> {
    inner: T,
    prefix: String,
    separator: String,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<T, Deps> PrefixedToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
{
    /// Create a new prefixed toolset with default separator "_".
    pub fn new(inner: T, prefix: impl Into<String>) -> Self {
        Self {
            inner,
            prefix: prefix.into(),
            separator: "_".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Set a custom separator (default is "_").
    #[must_use]
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Get the prefix.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Get the separator.
    #[must_use]
    pub fn separator(&self) -> &str {
        &self.separator
    }

    /// Get the inner toolset.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Create the prefixed name.
    fn prefixed_name(&self, name: &str) -> String {
        format!("{}{}{}", self.prefix, self.separator, name)
    }

    /// Strip the prefix from a name.
    fn strip_prefix(&self, prefixed: &str) -> Option<String> {
        let prefix_with_sep = format!("{}{}", self.prefix, self.separator);
        prefixed
            .strip_prefix(&prefix_with_sep)
            .map(|s| s.to_string())
    }
}

#[async_trait]
impl<T, Deps> AbstractToolset<Deps> for PrefixedToolset<T, Deps>
where
    T: AbstractToolset<Deps>,
    Deps: Send + Sync,
{
    fn id(&self) -> Option<&str> {
        self.inner.id()
    }

    fn type_name(&self) -> &'static str {
        "PrefixedToolset"
    }

    fn label(&self) -> String {
        format!(
            "PrefixedToolset('{}{}', {})",
            self.prefix,
            self.separator,
            self.inner.label()
        )
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let inner_tools = self.inner.get_tools(ctx).await?;

        Ok(inner_tools
            .into_iter()
            .map(|(name, mut tool)| {
                let prefixed = self.prefixed_name(&name);
                // Update the tool definition with the prefixed name
                tool.tool_def.name = prefixed.clone();
                (prefixed, tool)
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
        // Strip prefix to get the original tool name
        let original_name = self.strip_prefix(name).ok_or_else(|| {
            ToolError::not_found(format!(
                "Tool '{}' does not have expected prefix '{}{}'",
                name, self.prefix, self.separator
            ))
        })?;

        // Create a modified tool with the original name
        let mut original_tool = tool.clone();
        original_tool.tool_def.name = original_name.clone();

        self.inner
            .call_tool(&original_name, args, ctx, &original_tool)
            .await
    }

    async fn enter(&self) -> Result<(), ToolError> {
        self.inner.enter().await
    }

    async fn exit(&self) -> Result<(), ToolError> {
        self.inner.exit().await
    }
}

impl<T: std::fmt::Debug, Deps> std::fmt::Debug for PrefixedToolset<T, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixedToolset")
            .field("prefix", &self.prefix)
            .field("separator", &self.separator)
            .field("inner", &self.inner)
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
            args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            let query = args["query"].as_str().unwrap_or("*");
            Ok(ToolReturn::text(format!("Searching for: {}", query)))
        }
    }

    #[test]
    fn test_prefixed_name() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let prefixed = PrefixedToolset::new(toolset, "web");

        assert_eq!(prefixed.prefixed_name("search"), "web_search");
    }

    #[test]
    fn test_strip_prefix() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let prefixed = PrefixedToolset::new(toolset, "web");

        assert_eq!(prefixed.strip_prefix("web_search"), Some("search".to_string()));
        assert_eq!(prefixed.strip_prefix("local_search"), None);
    }

    #[test]
    fn test_custom_separator() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let prefixed = PrefixedToolset::new(toolset, "web").with_separator("::");

        assert_eq!(prefixed.prefixed_name("search"), "web::search");
    }

    #[tokio::test]
    async fn test_prefixed_toolset_get_tools() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let prefixed = PrefixedToolset::new(toolset, "web");

        let ctx = RunContext::minimal("test");
        let tools = prefixed.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("web_search"));
        assert!(!tools.contains_key("search"));

        let tool = tools.get("web_search").unwrap();
        assert_eq!(tool.tool_def.name, "web_search");
    }

    #[tokio::test]
    async fn test_prefixed_toolset_call_tool() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let prefixed = PrefixedToolset::new(toolset, "web");

        let ctx = RunContext::minimal("test");
        let tools = prefixed.get_tools(&ctx).await.unwrap();
        let tool = tools.get("web_search").unwrap();

        let result = prefixed
            .call_tool(
                "web_search",
                serde_json::json!({"query": "rust"}),
                &ctx,
                tool,
            )
            .await
            .unwrap();

        assert!(result.as_text().unwrap().contains("rust"));
    }

    #[tokio::test]
    async fn test_prefixed_toolset_wrong_prefix() {
        let toolset = FunctionToolset::new().tool(SearchTool);
        let prefixed = PrefixedToolset::new(toolset, "web");

        let ctx = RunContext::minimal("test");
        let fake_tool = ToolsetTool::new(ToolDefinition::new("local_search", "Local search"));

        let result = prefixed
            .call_tool("local_search", serde_json::json!({}), &ctx, &fake_tool)
            .await;

        assert!(result.is_err());
    }
}
