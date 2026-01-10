//! Core toolset trait and related types.
//!
//! This module provides the `AbstractToolset` trait which defines the interface
//! for collections of tools with shared management and lifecycle.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolDefinition, ToolError, ToolReturn};
use std::collections::HashMap;

/// A tool that belongs to a toolset.
///
/// This wraps a tool definition with additional metadata about
/// the toolset it belongs to.
#[derive(Debug, Clone)]
pub struct ToolsetTool {
    /// The toolset that owns this tool.
    pub toolset_id: Option<String>,
    /// The tool definition.
    pub tool_def: ToolDefinition,
    /// Maximum retries for this tool.
    pub max_retries: u32,
}

impl ToolsetTool {
    /// Create a new toolset tool.
    #[must_use]
    pub fn new(tool_def: ToolDefinition) -> Self {
        Self {
            toolset_id: None,
            tool_def,
            max_retries: 3,
        }
    }

    /// Set the toolset ID.
    #[must_use]
    pub fn with_toolset_id(mut self, id: impl Into<String>) -> Self {
        self.toolset_id = Some(id.into());
        self
    }

    /// Set max retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Get the tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.tool_def.name
    }

    /// Get the tool description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.tool_def.description
    }
}

/// Information about a toolset for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsetInfo {
    /// Toolset identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Toolset type name.
    pub type_name: String,
    /// Number of tools.
    pub tool_count: usize,
    /// Tool names.
    pub tool_names: Vec<String>,
}

/// Abstract toolset trait - collection of tools with shared management.
///
/// Toolsets group related tools together and can provide:
/// - Shared lifecycle management (enter/exit)
/// - Common configuration
/// - Tool filtering and transformation
///
/// # Type Parameters
///
/// - `Deps`: The type of dependencies available to tools.
#[async_trait]
pub trait AbstractToolset<Deps = ()>: Send + Sync {
    /// Unique identifier for this toolset.
    fn id(&self) -> Option<&str>;

    /// Human-readable label for error messages.
    fn label(&self) -> String {
        let mut label = self.type_name().to_string();
        if let Some(id) = self.id() {
            label.push_str(&format!(" '{}'", id));
        }
        label
    }

    /// Type name for debugging.
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Hint for resolving name conflicts.
    fn tool_name_conflict_hint(&self) -> String {
        format!(
            "Rename the tool or use PrefixedToolset to avoid conflicts in {}.",
            self.label()
        )
    }

    /// Get all available tools.
    ///
    /// Returns a map of tool names to their definitions.
    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError>;

    /// Call a tool by name.
    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError>;

    /// Enter context (for resource setup).
    ///
    /// Called before the toolset is used. Override to set up resources.
    async fn enter(&self) -> Result<(), ToolError> {
        Ok(())
    }

    /// Exit context (for cleanup).
    ///
    /// Called when done using the toolset. Override to clean up resources.
    async fn exit(&self) -> Result<(), ToolError> {
        Ok(())
    }
}

/// Boxed toolset for dynamic dispatch.
pub type BoxedToolset<Deps> = Box<dyn AbstractToolset<Deps>>;

/// Result type for toolset operations.
pub type ToolsetResult<T> = Result<T, ToolError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toolset_tool() {
        let def = ToolDefinition::new("test", "Test tool");
        let tool = ToolsetTool::new(def)
            .with_toolset_id("my_toolset")
            .with_max_retries(5);

        assert_eq!(tool.name(), "test");
        assert_eq!(tool.toolset_id, Some("my_toolset".to_string()));
        assert_eq!(tool.max_retries, 5);
    }

    #[test]
    fn test_toolset_info_serde() {
        let info = ToolsetInfo {
            id: Some("test_id".to_string()),
            type_name: "TestToolset".to_string(),
            tool_count: 3,
            tool_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: ToolsetInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.id, parsed.id);
        assert_eq!(info.tool_count, parsed.tool_count);
    }
}
