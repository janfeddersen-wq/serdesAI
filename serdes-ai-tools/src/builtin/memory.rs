//! Memory tool for Anthropic-specific agent memory.
//!
//! This module provides a simple builtin tool type for referencing
//! Anthropic's memory feature at the API level.

use serde::{Deserialize, Serialize};

/// Builtin tool for agent memory (Anthropic-specific).
///
/// This is a simple marker struct used to indicate that the agent
/// should have access to Anthropic's memory capabilities.
///
/// # Example
///
/// ```
/// use serdes_ai_tools::builtin::MemoryTool;
///
/// let memory = MemoryTool::new();
/// assert_eq!(memory.kind, "memory");
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryTool {
    /// The kind identifier for this tool type.
    pub kind: String,
}

impl MemoryTool {
    /// The static kind identifier for memory tools.
    pub const KIND: &'static str = "memory";

    /// Create a new memory tool instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kind: Self::KIND.into(),
        }
    }

    /// Get the kind identifier.
    #[must_use]
    pub fn kind() -> &'static str {
        Self::KIND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tool_new() {
        let tool = MemoryTool::new();
        assert_eq!(tool.kind, "memory");
    }

    #[test]
    fn test_memory_tool_kind() {
        assert_eq!(MemoryTool::kind(), "memory");
    }

    #[test]
    fn test_memory_tool_default() {
        let tool = MemoryTool::default();
        // Default will have empty string, new() should be used for proper initialization
        assert_eq!(tool.kind, "");
    }

    #[test]
    fn test_memory_tool_serialize() {
        let tool = MemoryTool::new();
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"kind\":\"memory\""));
    }

    #[test]
    fn test_memory_tool_deserialize() {
        let json = r#"{"kind":"memory"}"#;
        let tool: MemoryTool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.kind, "memory");
    }
}
