//! Tool registry for managing multiple tools.
//!
//! This module provides the `ToolRegistry` type which allows registering,
//! looking up, and calling tools by name.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    definition::ToolDefinition,
    errors::ToolError,
    return_types::ToolReturn,
    tool::Tool,
    RunContext,
};

/// Registry of tools that can be called by an agent.
///
/// The `ToolRegistry` manages a collection of tools and provides:
/// - Registration of tools
/// - Lookup by name
/// - Batch retrieval of definitions
/// - Tool execution
///
/// # Type Parameters
///
/// - `Deps`: The type of dependencies available to tools.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_tools::{ToolRegistry, Tool};
///
/// let mut registry = ToolRegistry::<()>::new();
/// registry.register(MyTool::new());
///
/// // Get all definitions for the model
/// let defs = registry.definitions();
///
/// // Call a tool by name
/// let result = registry.call("my_tool", &ctx, args).await?;
/// ```
pub struct ToolRegistry<Deps = ()> {
    tools: HashMap<String, Arc<dyn Tool<Deps>>>,
}

impl<Deps> ToolRegistry<Deps> {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    ///
    /// Returns `&mut self` for chaining.
    ///
    /// # Panics
    ///
    /// Panics if a tool with the same name is already registered.
    /// Use `register_replace` to allow replacement.
    pub fn register<T: Tool<Deps> + 'static>(&mut self, tool: T) -> &mut Self {
        let name = tool.definition().name.clone();
        if self.tools.contains_key(&name) {
            panic!("Tool '{}' is already registered", name);
        }
        self.tools.insert(name, Arc::new(tool));
        self
    }

    /// Register a tool, replacing any existing tool with the same name.
    pub fn register_replace<T: Tool<Deps> + 'static>(&mut self, tool: T) -> &mut Self {
        let name = tool.definition().name.clone();
        self.tools.insert(name, Arc::new(tool));
        self
    }

    /// Register a boxed tool.
    pub fn register_boxed(&mut self, tool: Arc<dyn Tool<Deps>>) -> &mut Self {
        let name = tool.definition().name.clone();
        self.tools.insert(name, tool);
        self
    }

    /// Register a tool if not already present.
    ///
    /// Returns `true` if the tool was registered, `false` if it already existed.
    pub fn register_if_absent<T: Tool<Deps> + 'static>(&mut self, tool: T) -> bool {
        let name = tool.definition().name.clone();
        if self.tools.contains_key(&name) {
            false
        } else {
            self.tools.insert(name, Arc::new(tool));
            true
        }
    }

    /// Unregister a tool by name.
    ///
    /// Returns the removed tool if it existed.
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn Tool<Deps>>> {
        self.tools.remove(name)
    }

    /// Get all tool definitions.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Get tool definitions as a map by name.
    #[must_use]
    pub fn definitions_map(&self) -> HashMap<String, ToolDefinition> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.definition()))
            .collect()
    }

    /// Get definitions with prepare applied.
    ///
    /// This calls the `prepare` method on each tool with the given context,
    /// allowing tools to modify their definitions or hide themselves.
    pub async fn prepared_definitions(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Vec<ToolDefinition>
    where
        Deps: Send + Sync,
    {
        let mut defs = Vec::with_capacity(self.tools.len());
        for tool in self.tools.values() {
            let base_def = tool.definition();
            if let Some(prepared) = tool.prepare(ctx, base_def).await {
                defs.push(prepared);
            }
        }
        defs
    }

    /// Call a tool by name.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::NotFound` if no tool with the given name exists.
    pub async fn call(
        &self,
        name: &str,
        ctx: &RunContext<Deps>,
        args: serde_json::Value,
    ) -> Result<ToolReturn, ToolError>
    where
        Deps: Send + Sync,
    {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::not_found(name))?;

        tool.call(ctx, args).await
    }

    /// Check if a tool exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool<Deps>>> {
        self.tools.get(name)
    }

    /// Get all tool names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Iterate over tools.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<dyn Tool<Deps>>)> {
        self.tools.iter()
    }

    /// Merge another registry into this one.
    ///
    /// Tools from `other` will replace tools with the same name in `self`.
    pub fn merge(&mut self, other: ToolRegistry<Deps>) {
        self.tools.extend(other.tools);
    }

    /// Clear all registered tools.
    pub fn clear(&mut self) {
        self.tools.clear();
    }

    /// Get max retries for a tool.
    #[must_use]
    pub fn max_retries(&self, name: &str) -> Option<u32> {
        self.tools.get(name).and_then(|t| t.max_retries())
    }
}

impl<Deps> Default for ToolRegistry<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Deps> std::fmt::Debug for ToolRegistry<Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.names())
            .finish()
    }
}

impl<Deps> Clone for ToolRegistry<Deps> {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

/// Trait for types that can provide tools to a registry.
pub trait ToolProvider<Deps> {
    /// Register tools with the given registry.
    fn register_tools(&self, registry: &mut ToolRegistry<Deps>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::SchemaBuilder, ToolResult};
    use async_trait::async_trait;

    struct EchoTool;

    #[async_trait]
    impl Tool<()> for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "Echo the message").with_parameters(
                SchemaBuilder::new()
                    .string("message", "Message", true)
                    .build()
                    .expect("SchemaBuilder JSON serialization failed"),
            )
        }

        async fn call(&self, _ctx: &RunContext<()>, args: serde_json::Value) -> ToolResult {
            let msg = args["message"].as_str().unwrap_or("<empty>");
            Ok(ToolReturn::text(msg))
        }
    }

    struct AddTool;

    #[async_trait]
    impl Tool<()> for AddTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("add", "Add two numbers")
        }

        async fn call(&self, _ctx: &RunContext<()>, args: serde_json::Value) -> ToolResult {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(ToolReturn::text(format!("{}", a + b)))
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::<()>::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("echo"));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_register_duplicate_panics() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(EchoTool); // Should panic
    }

    #[test]
    fn test_register_replace() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register_replace(EchoTool); // Should not panic
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_register_if_absent() {
        let mut registry = ToolRegistry::new();
        assert!(registry.register_if_absent(EchoTool));
        assert!(!registry.register_if_absent(EchoTool));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_unregister() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let removed = registry.unregister("echo");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(AddTool);

        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);
        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"add"));
    }

    #[tokio::test]
    async fn test_call() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let ctx = RunContext::minimal("test");
        let result = registry
            .call("echo", &ctx, serde_json::json!({"message": "hello"}))
            .await
            .unwrap();
        assert_eq!(result.as_text(), Some("hello"));
    }

    #[tokio::test]
    async fn test_call_not_found() {
        let registry = ToolRegistry::<()>::new();
        let ctx = RunContext::minimal("test");
        let result = registry.call("nonexistent", &ctx, serde_json::json!({})).await;
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }

    #[test]
    fn test_get() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_names() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(AddTool);

        let names = registry.names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_merge() {
        let mut registry1 = ToolRegistry::new();
        registry1.register(EchoTool);

        let mut registry2 = ToolRegistry::new();
        registry2.register(AddTool);

        registry1.merge(registry2);
        assert_eq!(registry1.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.clear();
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_prepared_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let ctx = RunContext::minimal("test");
        let prepared = registry.prepared_definitions(&ctx).await;
        assert_eq!(prepared.len(), 1);
    }

    #[test]
    fn test_clone() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);

        let cloned = registry.clone();
        assert_eq!(cloned.len(), registry.len());
    }

    #[test]
    fn test_debug() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let debug = format!("{:?}", registry);
        assert!(debug.contains("ToolRegistry"));
        assert!(debug.contains("echo"));
    }
}
