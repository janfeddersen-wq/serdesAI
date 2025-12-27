//! Combined toolset for merging multiple toolsets.
//!
//! This module provides `CombinedToolset`, which merges multiple toolsets
//! into a single unified toolset.

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use serdes_ai_tools::{RunContext, ToolError, ToolReturn};
use std::collections::HashMap;

use crate::{AbstractToolset, BoxedToolset, ToolsetTool};

/// Combines multiple toolsets into one.
///
/// This allows treating multiple toolsets as a single collection.
/// It handles name conflicts by returning an error with helpful hints.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_toolsets::{CombinedToolset, FunctionToolset};
///
/// let toolset1 = FunctionToolset::with_id("tools1").tool(tool_a);
/// let toolset2 = FunctionToolset::with_id("tools2").tool(tool_b);
///
/// let combined = CombinedToolset::new()
///     .add(toolset1)
///     .add(toolset2);
/// ```
pub struct CombinedToolset<Deps = ()> {
    id: Option<String>,
    toolsets: Vec<BoxedToolset<Deps>>,
}

impl<Deps> CombinedToolset<Deps> {
    /// Create a new empty combined toolset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: None,
            toolsets: Vec::new(),
        }
    }

    /// Create a combined toolset with an ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            toolsets: Vec::new(),
        }
    }

    /// Add a toolset.
    #[must_use]
    pub fn add<T: AbstractToolset<Deps> + 'static>(mut self, toolset: T) -> Self {
        self.toolsets.push(Box::new(toolset));
        self
    }

    /// Add a boxed toolset.
    #[must_use]
    pub fn add_boxed(mut self, toolset: BoxedToolset<Deps>) -> Self {
        self.toolsets.push(toolset);
        self
    }

    /// Add multiple toolsets.
    #[must_use]
    pub fn toolsets<I, T>(mut self, toolsets: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AbstractToolset<Deps> + 'static,
    {
        for toolset in toolsets {
            self.toolsets.push(Box::new(toolset));
        }
        self
    }

    /// Get the number of contained toolsets.
    #[must_use]
    pub fn toolset_count(&self) -> usize {
        self.toolsets.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.toolsets.is_empty()
    }
}

impl<Deps> Default for CombinedToolset<Deps> {
    fn default() -> Self {
        Self::new()
    }
}

/// Track which toolset owns which tool.
#[derive(Clone)]
struct ToolOwnership {
    toolset_index: usize,
    tool: ToolsetTool,
}

#[async_trait]
impl<Deps: Send + Sync + 'static> AbstractToolset<Deps> for CombinedToolset<Deps> {
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn type_name(&self) -> &'static str {
        "CombinedToolset"
    }

    fn tool_name_conflict_hint(&self) -> String {
        "Use PrefixedToolset to add prefixes to tool names from different toolsets.".to_string()
    }

    async fn get_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        let mut all_tools: HashMap<String, ToolOwnership> = HashMap::new();
        let mut conflicts: Vec<(String, String, String)> = Vec::new();

        for (idx, toolset) in self.toolsets.iter().enumerate() {
            let tools = toolset.get_tools(ctx).await?;

            for (name, tool) in tools {
                if let Some(existing) = all_tools.get(&name) {
                    // Track conflict
                    let existing_label = self.toolsets[existing.toolset_index].label();
                    let new_label = toolset.label();
                    conflicts.push((name.clone(), existing_label, new_label));
                } else {
                    all_tools.insert(
                        name,
                        ToolOwnership {
                            toolset_index: idx,
                            tool,
                        },
                    );
                }
            }
        }

        if !conflicts.is_empty() {
            let conflict_msgs: Vec<String> = conflicts
                .iter()
                .map(|(name, t1, t2)| format!("  - '{}' exists in {} and {}", name, t1, t2))
                .collect();

            return Err(ToolError::execution_failed(format!(
                "Tool name conflicts in {}:\n{}\n\nHint: {}",
                self.label(),
                conflict_msgs.join("\n"),
                self.tool_name_conflict_hint()
            )));
        }

        Ok(all_tools
            .into_iter()
            .map(|(name, ownership)| (name, ownership.tool))
            .collect())
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        ctx: &RunContext<Deps>,
        tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        // Find which toolset has this tool
        for toolset in &self.toolsets {
            let tools = toolset.get_tools(ctx).await?;
            if tools.contains_key(name) {
                return toolset.call_tool(name, args, ctx, tool).await;
            }
        }

        Err(ToolError::not_found(format!(
            "Tool '{}' not found in {}",
            name,
            self.label()
        )))
    }

    async fn enter(&self) -> Result<(), ToolError> {
        for toolset in &self.toolsets {
            toolset.enter().await?;
        }
        Ok(())
    }

    async fn exit(&self) -> Result<(), ToolError> {
        // Exit in reverse order
        for toolset in self.toolsets.iter().rev() {
            toolset.exit().await?;
        }
        Ok(())
    }
}

impl<Deps> std::fmt::Debug for CombinedToolset<Deps> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedToolset")
            .field("id", &self.id)
            .field("toolset_count", &self.toolsets.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionToolset;
    use async_trait::async_trait;
    use serdes_ai_tools::{ObjectJsonSchema, PropertySchema, Tool, ToolDefinition};

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

    struct ToolB;

    #[async_trait]
    impl Tool<()> for ToolB {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("tool_b", "Tool B")
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("B"))
        }
    }

    // Tool with same name as ToolA for conflict testing
    struct ConflictingTool;

    #[async_trait]
    impl Tool<()> for ConflictingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("tool_a", "Conflicting Tool A") // Same name!
        }

        async fn call(
            &self,
            _ctx: &RunContext<()>,
            _args: JsonValue,
        ) -> Result<ToolReturn, ToolError> {
            Ok(ToolReturn::text("Conflict"))
        }
    }

    #[test]
    fn test_combined_toolset_new() {
        let toolset = CombinedToolset::<()>::new();
        assert!(toolset.is_empty());
        assert_eq!(toolset.toolset_count(), 0);
    }

    #[test]
    fn test_combined_toolset_with_id() {
        let toolset = CombinedToolset::<()>::with_id("combined");
        assert_eq!(toolset.id(), Some("combined"));
    }

    #[tokio::test]
    async fn test_combined_toolset_merges_tools() {
        let ts1 = FunctionToolset::new().with_id("ts1").tool(ToolA);
        let ts2 = FunctionToolset::new().with_id("ts2").tool(ToolB);

        let combined = CombinedToolset::new().add(ts1).add(ts2);

        let ctx = RunContext::minimal("test");
        let tools = combined.get_tools(&ctx).await.unwrap();

        assert_eq!(tools.len(), 2);
        assert!(tools.contains_key("tool_a"));
        assert!(tools.contains_key("tool_b"));
    }

    #[tokio::test]
    async fn test_combined_toolset_call_tool() {
        let ts1 = FunctionToolset::new().tool(ToolA);
        let ts2 = FunctionToolset::new().tool(ToolB);

        let combined = CombinedToolset::new().add(ts1).add(ts2);

        let ctx = RunContext::minimal("test");
        let tools = combined.get_tools(&ctx).await.unwrap();
        let tool_a = tools.get("tool_a").unwrap();

        let result = combined
            .call_tool("tool_a", serde_json::json!({}), &ctx, tool_a)
            .await
            .unwrap();

        assert_eq!(result.as_text(), Some("A"));
    }

    #[tokio::test]
    async fn test_combined_toolset_conflict_detection() {
        let ts1 = FunctionToolset::new().with_id("ts1").tool(ToolA);
        let ts2 = FunctionToolset::new().with_id("ts2").tool(ConflictingTool);

        let combined = CombinedToolset::new().add(ts1).add(ts2);

        let ctx = RunContext::minimal("test");
        let result = combined.get_tools(&ctx).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("conflict"));
        assert!(err.message().contains("tool_a"));
    }

    #[tokio::test]
    async fn test_combined_toolset_enter_exit() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let enter_count = Arc::new(AtomicU32::new(0));
        let exit_count = Arc::new(AtomicU32::new(0));

        struct TrackedToolset {
            enter_count: Arc<AtomicU32>,
            exit_count: Arc<AtomicU32>,
        }

        #[async_trait]
        impl AbstractToolset<()> for TrackedToolset {
            fn id(&self) -> Option<&str> {
                None
            }

            async fn get_tools(
                &self,
                _ctx: &RunContext<()>,
            ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
                Ok(HashMap::new())
            }

            async fn call_tool(
                &self,
                _name: &str,
                _args: JsonValue,
                _ctx: &RunContext<()>,
                _tool: &ToolsetTool,
            ) -> Result<ToolReturn, ToolError> {
                Ok(ToolReturn::empty())
            }

            async fn enter(&self) -> Result<(), ToolError> {
                self.enter_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }

            async fn exit(&self) -> Result<(), ToolError> {
                self.exit_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let ts1 = TrackedToolset {
            enter_count: enter_count.clone(),
            exit_count: exit_count.clone(),
        };
        let ts2 = TrackedToolset {
            enter_count: enter_count.clone(),
            exit_count: exit_count.clone(),
        };

        let combined = CombinedToolset::new().add(ts1).add(ts2);

        combined.enter().await.unwrap();
        assert_eq!(enter_count.load(Ordering::SeqCst), 2);

        combined.exit().await.unwrap();
        assert_eq!(exit_count.load(Ordering::SeqCst), 2);
    }
}
