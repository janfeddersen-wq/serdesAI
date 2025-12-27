//! Function-based toolset implementation.
//!
//! This module provides `FunctionToolset`, which wraps a `ToolRegistry`
//! and adapts it to the `AbstractToolset` interface.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{
    ObjectJsonSchema, RunContext, Tool, ToolDefinition, ToolError, ToolRegistry, ToolReturn,
};
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::{AbstractToolset, ToolsetTool};

/// A toolset backed by function-based tools.
///
/// This wraps a `ToolRegistry` and provides the `AbstractToolset` interface.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::FunctionToolset;
/// use serdes_ai_tools::{SyncFunctionTool, ObjectJsonSchema};
///
/// let toolset = FunctionToolset::new()
///     .with_id("my_tools")
///     .tool(my_tool)
///     .tool(another_tool);
/// ```
pub struct FunctionToolset<Deps = ()> {
    id: Option<String>,
    registry: ToolRegistry<Deps>,
    max_retries: u32,
}

impl<Deps> FunctionToolset<Deps> {
    /// Create a new empty function toolset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            registry: ToolRegistry::new(),
            max_retries: 3,
        }
    }

    /// Set the toolset ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the default max retries for tools.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Add a tool to the toolset.
    #[must_use]
    pub fn tool<T: Tool<Deps> + 'static>(mut self, tool: T) -> Self {
        self.registry.register(tool);
        self
    }

    /// Add multiple tools to the toolset.
    #[must_use]
    pub fn tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Tool<Deps> + 'static,
    {
        for tool in tools {
            self.registry.register(tool);
        }
        self
    }

    /// Get the underlying registry.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry<Deps> {
        &self.registry
    }

    /// Get mutable access to the registry.
    pub fn registry_mut(&mut self) -> &mut ToolRegistry<Deps> {
        &mut self.registry
    }

    /// Get the number of tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Check if the toolset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }
}

impl<Deps> Default for FunctionToolset<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync + 'static> AbstractToolset<Deps> for FunctionToolset<Deps> {
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn type_name(&self) -> &'static str {
        "FunctionToolset"
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let defs = self.registry.prepared_definitions(ctx).await;
        Ok(defs
            .into_iter()
            .map(|def| {
                let name = def.name.clone();
                let max_retries = self
                    .registry
                    .max_retries(&name)
                    .unwrap_or(self.max_retries);
                (
                    name,
                    ToolsetTool {
                        toolset_id: self.id.clone(),
                        tool_def: def,
                        max_retries,
                    },
                )
            })
            .collect())
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        _tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        self.registry.call(name, ctx, args).await
    }
}

impl<Deps> std::fmt::Debug for FunctionToolset<Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionToolset")
            .field("id", &self.id)
            .field("tool_count", &self.registry.len())
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// Wrapper for async function tools with explicit type.
///
/// This provides a simpler way to add async functions as tools.
pub struct AsyncFnTool<F, Deps> {
    name: String,
    description: String,
    parameters: ObjectJsonSchema,
    function: F,
    max_retries: Option<u32>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<F, Deps> AsyncFnTool<F, Deps> {
    /// Create a new async function tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: ObjectJsonSchema,
        function: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            function,
            max_retries: None,
            _phantom: PhantomData,
        }
    }

    /// Set max retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }
}

type PinnedToolFuture = Pin<Box<dyn Future<Output = Result<ToolReturn, ToolError>> + Send>>;

#[async_trait]
impl<F, Deps> Tool<Deps> for AsyncFnTool<F, Deps>
where
    F: for<'a> Fn(&'a RunContext<Deps>, JsonValue) -> PinnedToolFuture + Send + Sync,
    Deps: Send + Sync,
{
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(&self.name, &self.description)
            .with_parameters(self.parameters.clone())
    }

    async fn call(
        &self,
        ctx: &RunContext<Deps>,
        args: JsonValue,
    ) -> Result<ToolReturn, ToolError> {
        (self.function)(ctx, args).await
    }

    fn max_retries(&self) -> Option<u32> {
        self.max_retries
    }
}

impl<F, Deps> std::fmt::Debug for AsyncFnTool<F, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncFnTool")
            .field("name", &self.name)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_tools::{PropertySchema, SyncFunctionTool};

    struct EchoTool;

    #[async_trait]
    impl Tool<()> for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "Echo the message")
                .with_parameters(
                    ObjectJsonSchema::new()
                        .with_property("msg", PropertySchema::string("Message").build(), true),
                )
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            let msg = args["msg"].as_str().unwrap_or("<none>");
            Ok(ToolReturn::text(msg))
        }
    }

    #[test]
    fn test_function_toolset_new() {
        let toolset = FunctionToolset::<()>::new();
        assert!(toolset.is_empty());
        assert!(toolset.id().is_none());
    }

    #[test]
    fn test_function_toolset_with_id() {
        let toolset = FunctionToolset::<()>::new().with_id("my_tools");
        assert_eq!(toolset.id(), Some("my_tools"));
    }

    #[test]
    fn test_function_toolset_add_tool() {
        let toolset = FunctionToolset::new().tool(EchoTool);
        assert_eq!(toolset.len(), 1);
    }

    #[tokio::test]
    async fn test_function_toolset_get_tools() {
        let toolset: FunctionToolset<()> = FunctionToolset::new()
            .with_id("test")
            .tool(EchoTool);

        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("echo"));
        let echo = tools.get("echo").unwrap();
        assert_eq!(echo.toolset_id, Some("test".to_string()));
    }

    #[tokio::test]
    async fn test_function_toolset_call_tool() {
        let toolset = FunctionToolset::new().tool(EchoTool);
        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();
        let echo_tool = tools.get("echo").unwrap();

        let result = toolset
            .call_tool("echo", serde_json::json!({"msg": "hello"}), &ctx, echo_tool)
            .await
            .unwrap();

        assert_eq!(result.as_text(), Some("hello"));
    }

    #[test]
    fn test_function_toolset_debug() {
        let toolset = FunctionToolset::new()
            .with_id("debug_test")
            .with_max_retries(5)
            .tool(EchoTool);

        let debug = format!("{:?}", toolset);
        assert!(debug.contains("FunctionToolset"));
        assert!(debug.contains("debug_test"));
    }
}
