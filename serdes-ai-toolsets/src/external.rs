//! External toolset implementation.
//!
//! This module provides `ExternalToolset`, for tools that are executed
//! externally (not by the agent itself).

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolDefinition, ToolError, ToolReturn};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{AbstractToolset, ToolsetTool};

/// Toolset for externally-executed tools.
///
/// This is used when tools need to be exposed to the model but will be
/// executed by an external system. When any tool is called, it returns
/// `ToolError::CallDeferred` so the agent knows to defer execution.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::ExternalToolset;
/// use serdes_ai_tools::ToolDefinition;
///
/// let external = ExternalToolset::new()
///     .with_id("external_api")
///     .definition(ToolDefinition::new("api_call", "Call external API"));
/// ```
pub struct ExternalToolset<Deps = ()> {
    id: Option<String>,
    definitions: Vec<ToolDefinition>,
    max_retries: u32,
    _phantom: PhantomData<fn() -> Deps>,
}

impl<Deps> ExternalToolset<Deps> {
    /// Create a new empty external toolset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            definitions: Vec::new(),
            max_retries: 3,
            _phantom: PhantomData,
        }
    }

    /// Set the toolset ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set max retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Add a tool definition.
    #[must_use]
    pub fn definition(mut self, def: ToolDefinition) -> Self {
        self.definitions.push(def);
        self
    }

    /// Add multiple tool definitions.
    #[must_use]
    pub fn definitions(mut self, defs: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.definitions.extend(defs);
        self
    }

    /// Get the number of definitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}

impl<Deps> Default for ExternalToolset<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> AbstractToolset<Deps> for ExternalToolset<Deps> {
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn type_name(&self) -> &'static str {
        "ExternalToolset"
    }

    async fn get_tools(
        &self,
        _ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        Ok(self
            .definitions
            .iter()
            .map(|def| {
                (
                    def.name.clone(),
                    ToolsetTool {
                        toolset_id: self.id.clone(),
                        tool_def: def.clone(),
                        max_retries: self.max_retries,
                    },
                )
            })
            .collect())
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        _ctx: &RunContext<Deps>,
        _tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        // Always defer external tool calls
        Err(ToolError::CallDeferred {
            tool_name: name.to_string(),
            args,
        })
    }
}

impl<Deps> std::fmt::Debug for ExternalToolset<Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalToolset")
            .field("id", &self.id)
            .field("definitions", &self.definitions.len())
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_toolset_new() {
        let toolset = ExternalToolset::<()>::new();
        assert!(toolset.is_empty());
        assert!(toolset.id().is_none());
    }

    #[test]
    fn test_external_toolset_with_definitions() {
        let toolset = ExternalToolset::<()>::new()
            .with_id("external")
            .definition(ToolDefinition::new("api_call", "Call API"))
            .definition(ToolDefinition::new("webhook", "Send webhook"));

        assert_eq!(toolset.len(), 2);
        assert_eq!(toolset.id(), Some("external"));
    }

    #[tokio::test]
    async fn test_external_toolset_get_tools() {
        let toolset = ExternalToolset::<()>::new()
            .definition(ToolDefinition::new("test", "Test tool"));

        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 1);
        assert!(tools.contains_key("test"));
    }

    #[tokio::test]
    async fn test_external_toolset_call_deferred() {
        let toolset = ExternalToolset::<()>::new()
            .definition(ToolDefinition::new("api_call", "Call API"));

        let ctx = RunContext::minimal("test");
        let tools = toolset.get_tools(&ctx).await.unwrap();
        let tool = tools.get("api_call").unwrap();

        let result = toolset
            .call_tool("api_call", serde_json::json!({"endpoint": "/test"}), &ctx, tool)
            .await;

        assert!(matches!(result, Err(ToolError::CallDeferred { .. })));

        if let Err(ToolError::CallDeferred { tool_name, args }) = result {
            assert_eq!(tool_name, "api_call");
            assert_eq!(args["endpoint"], "/test");
        }
    }
}
