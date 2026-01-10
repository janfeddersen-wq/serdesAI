//! Dynamic toolset implementation.
//!
//! This module provides `DynamicToolset`, which allows tools to be
//! added and removed at runtime.

use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, Tool, ToolError, ToolReturn};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{AbstractToolset, ToolsetTool};

/// Toolset that can have tools added/removed at runtime.
///
/// This is useful for scenarios where the available tools change
/// during the agent's lifetime.
///
/// # Thread Safety
///
/// All operations are thread-safe and can be called concurrently.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::DynamicToolset;
///
/// let toolset = DynamicToolset::new();
///
/// // Add tools at runtime
/// toolset.add_tool(my_tool);
///
/// // Remove tools
/// toolset.remove_tool("my_tool");
/// ```
pub struct DynamicToolset<Deps = ()>
where
    Deps: Send + Sync + 'static,
{
    id: Option<String>,
    tools: RwLock<HashMap<String, Arc<dyn Tool<Deps>>>>,
    max_retries: u32,
}

impl<Deps: Send + Sync + 'static> DynamicToolset<Deps> {
    /// Create a new empty dynamic toolset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            tools: RwLock::new(HashMap::new()),
            max_retries: 3,
        }
    }

    /// Create with an ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            tools: RwLock::new(HashMap::new()),
            max_retries: 3,
        }
    }

    /// Set max retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Add a tool.
    ///
    /// If a tool with the same name exists, it will be replaced.
    pub fn add_tool<T: Tool<Deps> + 'static>(&self, tool: T) {
        let name = tool.definition().name.clone();
        self.tools.write().insert(name, Arc::new(tool));
    }

    /// Add a boxed tool.
    pub fn add_boxed(&self, tool: Arc<dyn Tool<Deps>>) {
        let name = tool.definition().name.clone();
        self.tools.write().insert(name, tool);
    }

    /// Remove a tool by name.
    ///
    /// Returns `true` if the tool was removed, `false` if it didn't exist.
    pub fn remove_tool(&self, name: &str) -> bool {
        self.tools.write().remove(name).is_some()
    }

    /// Clear all tools.
    pub fn clear(&self) {
        self.tools.write().clear();
    }

    /// Get the number of tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.read().len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.read().is_empty()
    }

    /// Check if a tool exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.read().contains_key(name)
    }

    /// Get tool names.
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }
}

impl<Deps: Send + Sync + 'static> Default for DynamicToolset<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync + 'static> AbstractToolset<Deps> for DynamicToolset<Deps> {
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn type_name(&self) -> &'static str {
        "DynamicToolset"
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        // Clone the tools under the lock to avoid holding it across await
        let tools_snapshot: Vec<(String, Arc<dyn Tool<Deps>>)> = {
            let tools = self.tools.read();
            tools
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(v)))
                .collect()
        };

        let mut result = HashMap::with_capacity(tools_snapshot.len());

        for (name, tool) in tools_snapshot {
            let def = tool.definition();

            // Apply prepare if available
            let prepared_def = tool.prepare(ctx, def.clone()).await;

            if let Some(final_def) = prepared_def {
                let max_retries = tool.max_retries().unwrap_or(self.max_retries);
                result.insert(
                    name,
                    ToolsetTool {
                        toolset_id: self.id.clone(),
                        tool_def: final_def,
                        max_retries,
                    },
                );
            }
        }

        Ok(result)
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        _tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        let tool = {
            let tools = self.tools.read();
            tools
                .get(name)
                .cloned()
                .ok_or_else(|| ToolError::not_found(name))?
        };

        tool.call(ctx, args).await
    }
}

impl<Deps: Send + Sync + 'static> std::fmt::Debug for DynamicToolset<Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicToolset")
            .field("id", &self.id)
            .field("tool_count", &self.len())
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_tools::ToolDefinition;

    struct EchoTool {
        prefix: String,
    }

    impl EchoTool {
        fn new(prefix: impl Into<String>) -> Self {
            Self {
                prefix: prefix.into(),
            }
        }
    }

    #[async_trait]
    impl Tool<()> for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "Echo with prefix")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            let msg = args["msg"].as_str().unwrap_or("<none>");
            Ok(ToolReturn::text(format!("{}{}", self.prefix, msg)))
        }
    }

    struct AddTool;

    #[async_trait]
    impl Tool<()> for AddTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("add", "Add numbers")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok(ToolReturn::text(format!("{}", a + b)))
        }
    }

    #[test]
    fn test_dynamic_toolset_new() {
        let toolset = DynamicToolset::<()>::new();
        assert!(toolset.is_empty());
    }

    #[test]
    fn test_dynamic_toolset_add_remove() {
        let toolset = DynamicToolset::<()>::new();

        toolset.add_tool(EchoTool::new(">>> "));
        assert_eq!(toolset.len(), 1);
        assert!(toolset.contains("echo"));

        toolset.add_tool(AddTool);
        assert_eq!(toolset.len(), 2);

        assert!(toolset.remove_tool("echo"));
        assert_eq!(toolset.len(), 1);
        assert!(!toolset.contains("echo"));

        assert!(!toolset.remove_tool("nonexistent"));
    }

    #[test]
    fn test_dynamic_toolset_clear() {
        let toolset = DynamicToolset::<()>::new();
        toolset.add_tool(EchoTool::new(""));
        toolset.add_tool(AddTool);

        toolset.clear();
        assert!(toolset.is_empty());
    }

    #[test]
    fn test_dynamic_toolset_tool_names() {
        let toolset = DynamicToolset::<()>::new();
        toolset.add_tool(EchoTool::new(""));
        toolset.add_tool(AddTool);

        let names = toolset.tool_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"echo".to_string()));
        assert!(names.contains(&"add".to_string()));
    }

    #[tokio::test]
    async fn test_dynamic_toolset_get_tools() {
        let toolset = DynamicToolset::<()>::new();
        toolset.add_tool(EchoTool::new(""));

        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("echo"));
    }

    #[tokio::test]
    async fn test_dynamic_toolset_call_tool() {
        let toolset = DynamicToolset::<()>::new();
        toolset.add_tool(EchoTool::new("[PREFIX] "));

        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();
        let tool = tools.get("echo").unwrap();

        let result = toolset
            .call_tool("echo", serde_json::json!({"msg": "hello"}), &ctx, tool)
            .await
            .unwrap();

        assert_eq!(result.as_text(), Some("[PREFIX] hello"));
    }

    #[tokio::test]
    async fn test_dynamic_toolset_replace_tool() {
        let toolset = DynamicToolset::<()>::new();
        toolset.add_tool(EchoTool::new("v1: "));

        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();
        let tool = tools.get("echo").unwrap();

        let result1 = toolset
            .call_tool("echo", serde_json::json!({"msg": "test"}), &ctx, tool)
            .await
            .unwrap();
        assert_eq!(result1.as_text(), Some("v1: test"));

        // Replace with v2
        toolset.add_tool(EchoTool::new("v2: "));

        let result2 = toolset
            .call_tool("echo", serde_json::json!({"msg": "test"}), &ctx, tool)
            .await
            .unwrap();
        assert_eq!(result2.as_text(), Some("v2: test"));
    }

    #[tokio::test]
    async fn test_dynamic_toolset_concurrent_access() {
        use std::sync::Arc;
        use tokio::task::JoinSet;

        let toolset = Arc::new(DynamicToolset::<()>::new());

        let mut tasks = JoinSet::new();

        // Spawn multiple tasks that add tools
        for i in 0..10 {
            let ts = toolset.clone();
            tasks.spawn(async move {
                ts.add_tool(EchoTool::new(format!("task{}: ", i)));
            });
        }

        // Wait for all to complete
        while tasks.join_next().await.is_some() {}

        // All tools should be there (but with possible overwrites for "echo")
        assert!(!toolset.is_empty());
    }
}
