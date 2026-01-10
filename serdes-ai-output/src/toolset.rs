//! Output toolset implementation.
//!
//! This module provides `OutputToolset`, an internal toolset that captures
//! structured output via tool calls.

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolError, ToolReturn};
use serdes_ai_toolsets::{AbstractToolset, ToolsetTool};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::schema::OutputSchema;
use crate::structured::StructuredOutputSchema;

/// Internal toolset that captures output via tool calls.
///
/// This toolset provides a special tool that the model calls to
/// "return" its structured output. The output is captured and
/// stored for retrieval.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_output::{OutputToolset, StructuredOutputSchema};
///
/// let schema = StructuredOutputSchema::<MyOutput>::new(my_json_schema);
/// let toolset = OutputToolset::new(schema);
///
/// // Add to agent's tools...
/// // After model calls the output tool, retrieve the captured output:
/// if let Some(output) = toolset.take_output() {
///     println!("Got output: {:?}", output);
/// }
/// ```
pub struct OutputToolset<T, Deps = ()>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    schema: StructuredOutputSchema<T>,
    captured: Arc<RwLock<Option<T>>>,
    _phantom: PhantomData<Deps>,
}

impl<T, Deps> OutputToolset<T, Deps>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    /// Create a new output toolset.
    pub fn new(schema: StructuredOutputSchema<T>) -> Self {
        Self {
            schema,
            captured: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        }
    }

    /// Check if output has been captured.
    #[must_use]
    pub fn has_output(&self) -> bool {
        self.captured.read().is_some()
    }

    /// Take the captured output, if any.
    pub fn take_output(&self) -> Option<T> {
        self.captured.write().take()
    }

    /// Get a reference to the captured output.
    pub fn get_output(&self) -> Option<T>
    where
        T: Clone,
    {
        self.captured.read().clone()
    }

    /// Clear any captured output.
    pub fn clear(&self) {
        *self.captured.write() = None;
    }

    /// Get the schema.
    #[must_use]
    pub fn schema(&self) -> &StructuredOutputSchema<T> {
        &self.schema
    }

    /// Get the tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.schema.tool_name
    }
}

impl<T, Deps> std::fmt::Debug for OutputToolset<T, Deps>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputToolset")
            .field("tool_name", &self.schema.tool_name)
            .field("has_output", &self.has_output())
            .finish()
    }
}

#[async_trait]
impl<T, Deps> AbstractToolset<Deps> for OutputToolset<T, Deps>
where
    T: DeserializeOwned + Serialize + Send + Sync + 'static,
    Deps: Send + Sync + 'static,
{
    fn id(&self) -> Option<&str> {
        Some("__output__")
    }

    fn type_name(&self) -> &'static str {
        "OutputToolset"
    }

    async fn get_tools(
        &self,
        _ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let defs = self.schema.tool_definitions();
        let mut tools = HashMap::with_capacity(defs.len());

        for def in defs {
            let name = def.name.clone();
            tools.insert(name, ToolsetTool::new(def).with_toolset_id("__output__"));
        }

        Ok(tools)
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        _ctx: &RunContext<Deps>,
        _tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        // Parse and capture the output
        let output: T = self
            .schema
            .parse_tool_call(name, &args)
            .map_err(|e| ToolError::execution_failed(e.to_string()))?;

        // Store the captured output
        *self.captured.write() = Some(output);

        // Return success with the args as confirmation
        Ok(ToolReturn::json(args))
    }
}

/// Marker type for output tool results.
#[derive(Debug, Clone)]
pub struct OutputCaptured<T> {
    /// The captured value.
    pub value: T,
    /// The tool name that was called.
    pub tool_name: String,
}

impl<T> OutputCaptured<T> {
    /// Create a new output captured marker.
    pub fn new(value: T, tool_name: impl Into<String>) -> Self {
        Self {
            value,
            tool_name: tool_name.into(),
        }
    }

    /// Unwrap the captured value.
    pub fn into_inner(self) -> T {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serdes_ai_tools::{ObjectJsonSchema, PropertySchema};

    #[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
    struct TestOutput {
        message: String,
        count: i32,
    }

    fn test_schema() -> StructuredOutputSchema<TestOutput> {
        let json_schema = ObjectJsonSchema::new()
            .with_property(
                "message",
                PropertySchema::string("The message").build(),
                true,
            )
            .with_property("count", PropertySchema::integer("The count").build(), true);

        StructuredOutputSchema::new(json_schema)
    }

    #[test]
    fn test_output_toolset_new() {
        let toolset: OutputToolset<TestOutput> = OutputToolset::new(test_schema());
        assert!(!toolset.has_output());
        assert_eq!(toolset.id(), Some("__output__"));
    }

    #[tokio::test]
    async fn test_output_toolset_get_tools() {
        let toolset: OutputToolset<TestOutput, ()> = OutputToolset::new(test_schema());
        let ctx = RunContext::minimal("test");

        let tools = toolset.get_tools(&ctx).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("final_result"));
    }

    #[tokio::test]
    async fn test_output_toolset_call_and_capture() {
        let toolset: OutputToolset<TestOutput, ()> = OutputToolset::new(test_schema());
        let ctx = RunContext::minimal("test");

        let tools = toolset.get_tools(&ctx).await.unwrap();
        let tool = tools.get("final_result").unwrap();

        let args = serde_json::json!({
            "message": "Hello, World!",
            "count": 42
        });

        let result = toolset.call_tool("final_result", args, &ctx, tool).await;
        assert!(result.is_ok());

        // Check captured output
        assert!(toolset.has_output());
        let output = toolset.take_output().unwrap();
        assert_eq!(output.message, "Hello, World!");
        assert_eq!(output.count, 42);

        // After take, should be empty
        assert!(!toolset.has_output());
    }

    #[tokio::test]
    async fn test_output_toolset_clear() {
        let toolset: OutputToolset<TestOutput, ()> = OutputToolset::new(test_schema());
        let ctx = RunContext::minimal("test");

        let tools = toolset.get_tools(&ctx).await.unwrap();
        let tool = tools.get("final_result").unwrap();

        let args = serde_json::json!({
            "message": "Test",
            "count": 1
        });

        toolset
            .call_tool("final_result", args, &ctx, tool)
            .await
            .unwrap();

        assert!(toolset.has_output());
        toolset.clear();
        assert!(!toolset.has_output());
    }

    #[tokio::test]
    async fn test_output_toolset_wrong_tool_name() {
        let toolset: OutputToolset<TestOutput, ()> = OutputToolset::new(test_schema());
        let ctx = RunContext::minimal("test");

        let tools = toolset.get_tools(&ctx).await.unwrap();
        let tool = tools.get("final_result").unwrap();

        let args = serde_json::json!({"message": "Test", "count": 1});

        let result = toolset.call_tool("wrong_name", args, &ctx, tool).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_output_captured() {
        let captured = OutputCaptured::new("test".to_string(), "my_tool");
        assert_eq!(captured.tool_name, "my_tool");
        assert_eq!(captured.into_inner(), "test");
    }
}
