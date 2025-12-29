//! Core tool trait and implementations.
//!
//! This module provides the `Tool` trait which all tools must implement,
//! as well as the `FunctionTool` wrapper for closure-based tools.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    definition::ToolDefinition,
    return_types::ToolResult,
    schema::SchemaBuilder,
    RunContext,
};

/// Core trait for all tools.
///
/// Implement this trait to create custom tools that can be called by agents.
/// Tools receive a context with dependencies and arguments as JSON.
///
/// # Type Parameters
///
/// - `Deps`: The type of dependencies the tool requires. Defaults to `()`.
///
/// # Example
///
/// ```ignore
/// use async_trait::async_trait;
/// use serdes_ai_tools::{Tool, ToolDefinition, RunContext, ToolResult, ToolReturn};
///
/// struct GreetTool;
///
/// #[async_trait]
/// impl Tool for GreetTool {
///     fn definition(&self) -> ToolDefinition {
///         ToolDefinition::new("greet", "Greet someone")
///     }
///
///     async fn call(&self, _ctx: &RunContext, args: serde_json::Value) -> ToolResult {
///         let name = args["name"].as_str().unwrap_or("World");
///         Ok(ToolReturn::text(format!("Hello, {name}!")))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool<Deps = ()>: Send + Sync {
    /// Get the tool's definition.
    ///
    /// This provides the name, description, and parameter schema that
    /// will be sent to the language model.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with given arguments.
    ///
    /// # Arguments
    ///
    /// - `ctx`: The run context with dependencies and metadata
    /// - `args`: The arguments as a JSON value
    ///
    /// # Returns
    ///
    /// A `ToolResult` containing either the return value or an error.
    async fn call(&self, ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult;

    /// Maximum retries for this tool.
    ///
    /// Returns `None` to use the agent's default retry setting.
    fn max_retries(&self) -> Option<u32> {
        None
    }

    /// Prepare the tool definition at runtime.
    ///
    /// This allows modifying the tool definition based on the current context,
    /// for example to add dynamic descriptions or modify parameters.
    ///
    /// Return `None` to indicate the tool should be hidden from this run.
    async fn prepare(
        &self,
        _ctx: &RunContext<Deps>,
        def: ToolDefinition,
    ) -> Option<ToolDefinition> {
        Some(def)
    }

    /// Get the tool name.
    fn name(&self) -> String {
        self.definition().name.clone()
    }

    /// Get the tool description.
    fn description(&self) -> String {
        self.definition().description.clone()
    }
}

/// Type-erased boxed tool.
pub type BoxedTool<Deps> = Arc<dyn Tool<Deps>>;

/// Wrapper for function-based tools.
///
/// This allows creating tools from async closures without implementing
/// the `Tool` trait manually.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_tools::{FunctionTool, SchemaBuilder, ToolReturn};
///
/// let tool = FunctionTool::new(
///     "add",
///     "Add two numbers",
///     SchemaBuilder::new()
///         .number("a", "First number", true)
///         .number("b", "Second number", true)
///         .build()
///         .unwrap(),
///     |_ctx, args| async move {
///         let a = args["a"].as_f64().unwrap_or(0.0);
///         let b = args["b"].as_f64().unwrap_or(0.0);
///         Ok(ToolReturn::text(format!("{}", a + b)))
///     },
/// );
/// ```
pub struct FunctionTool<F, Deps = ()> {
    name: String,
    description: String,
    parameters: JsonValue,
    function: F,
    max_retries: Option<u32>,
    strict: Option<bool>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<F, Deps> FunctionTool<F, Deps> {
    /// Create a new function tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Into<JsonValue>,
        function: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: parameters.into(),
            function,
            max_retries: None,
            strict: None,
            _phantom: PhantomData,
        }
    }

    /// Set maximum retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set strict mode.
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

// We need a type alias for the pinned future to make the bounds work
type PinnedFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

#[async_trait]
impl<F, Deps> Tool<Deps> for FunctionTool<F, Deps>
where
    F: for<'a> Fn(&'a RunContext<Deps>, JsonValue) -> PinnedFuture<ToolResult> + Send + Sync,
    Deps: Send + Sync,
{
    fn definition(&self) -> ToolDefinition {
        let mut def = ToolDefinition::new(&self.name, &self.description)
            .with_parameters(self.parameters.clone());
        if let Some(strict) = self.strict {
            def = def.with_strict(strict);
        }
        def
    }

    async fn call(&self, ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        (self.function)(ctx, args).await
    }

    fn max_retries(&self) -> Option<u32> {
        self.max_retries
    }
}

impl<F, Deps> std::fmt::Debug for FunctionTool<F, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// Wrapper for sync function tools.
///
/// For tools that don't need async, this provides a simpler API.
pub struct SyncFunctionTool<F, Deps = ()> {
    name: String,
    description: String,
    parameters: JsonValue,
    function: F,
    max_retries: Option<u32>,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<F, Deps> SyncFunctionTool<F, Deps>
where
    F: Fn(&RunContext<Deps>, JsonValue) -> ToolResult + Send + Sync,
{
    /// Create a new sync function tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Into<JsonValue>,
        function: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: parameters.into(),
            function,
            max_retries: None,
            _phantom: PhantomData,
        }
    }

    /// Set maximum retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }
}

#[async_trait]
impl<F, Deps> Tool<Deps> for SyncFunctionTool<F, Deps>
where
    F: Fn(&RunContext<Deps>, JsonValue) -> ToolResult + Send + Sync,
    Deps: Send + Sync,
{
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(&self.name, &self.description)
            .with_parameters(self.parameters.clone())
    }

    async fn call(&self, ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        (self.function)(ctx, args)
    }

    fn max_retries(&self) -> Option<u32> {
        self.max_retries
    }
}

impl<F, Deps> std::fmt::Debug for SyncFunctionTool<F, Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncFunctionTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// Create a simple sync tool from a function.
///
/// # Example
///
/// ```ignore
/// let tool = sync_tool(
///     "echo",
///     "Echo the input",
///     |_ctx, args| {
///         let msg = args["message"].as_str().unwrap_or("");
///         Ok(ToolReturn::text(msg))
///     },
/// );
/// ```
pub fn sync_tool<F, Deps>(
    name: impl Into<String>,
    description: impl Into<String>,
    function: F,
) -> SyncFunctionTool<F, Deps>
where
    F: Fn(&RunContext<Deps>, JsonValue) -> ToolResult + Send + Sync,
{
    SyncFunctionTool::new(
        name,
        description,
        SchemaBuilder::new()
            .build()
            .expect("SchemaBuilder JSON serialization failed"),
        function,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolReturn;

    #[derive(Debug, Clone, Default)]
    struct TestDeps;

    struct TestTool;

    #[async_trait]
    impl Tool<TestDeps> for TestTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("test", "Test tool").with_parameters(
                SchemaBuilder::new()
                    .integer("x", "A number", true)
                    .build()
                    .expect("SchemaBuilder JSON serialization failed"),
            )
        }

        async fn call(&self, _ctx: &RunContext<TestDeps>, args: JsonValue) -> ToolResult {
            let x = args["x"].as_i64().unwrap_or(0);
            Ok(ToolReturn::text(format!("x = {x}")))
        }

        fn max_retries(&self) -> Option<u32> {
            Some(5)
        }
    }

    #[tokio::test]
    async fn test_tool_trait() {
        let tool = TestTool;
        let ctx = RunContext::new(TestDeps, "test-model");

        assert_eq!(tool.name(), "test");
        assert_eq!(tool.description(), "Test tool");
        assert_eq!(tool.max_retries(), Some(5));

        let result = tool.call(&ctx, serde_json::json!({"x": 42})).await.unwrap();
        assert_eq!(result.as_text(), Some("x = 42"));
    }

    #[tokio::test]
    async fn test_sync_function_tool() {
        let tool = SyncFunctionTool::new(
            "add",
            "Add numbers",
            SchemaBuilder::new()
                .number("a", "First", true)
                .number("b", "Second", true)
                .build()
                .expect("SchemaBuilder JSON serialization failed"),
            |_ctx: &RunContext<()>, args: JsonValue| {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                Ok(ToolReturn::text(format!("{}", a + b)))
            },
        );

        let ctx = RunContext::minimal("test");
        let result = tool.call(&ctx, serde_json::json!({"a": 1.5, "b": 2.5})).await.unwrap();
        assert_eq!(result.as_text(), Some("4"));
    }

    #[tokio::test]
    async fn test_tool_prepare() {
        let tool = TestTool;
        let ctx = RunContext::new(TestDeps, "test");
        let def = tool.definition();
        let prepared = tool.prepare(&ctx, def.clone()).await;
        assert!(prepared.is_some());
        assert_eq!(prepared.unwrap().name, def.name);
    }

    #[test]
    fn test_sync_tool_helper() {
        let tool = sync_tool::<_, ()>(
            "echo",
            "Echo",
            |_ctx, args| {
                let msg = args["message"].as_str().unwrap_or("default");
                Ok(ToolReturn::text(msg))
            },
        );
        assert_eq!(tool.name, "echo");
    }
}