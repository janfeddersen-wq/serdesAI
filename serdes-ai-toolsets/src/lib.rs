//! # serdes-ai-toolsets
//!
//! Toolset abstractions for grouping and managing tools.
//!
//! This crate provides the infrastructure for organizing tools into logical
//! groups with shared configuration, lifecycle management, and composition.
//!
//! ## Core Concepts
//!
//! - **[`AbstractToolset`]**: Base trait for all toolsets
//! - **[`FunctionToolset`]**: Wrap function-based tools
//! - **[`CombinedToolset`]**: Merge multiple toolsets
//! - **[`DynamicToolset`]**: Runtime tool management
//!
//! ## Toolset Wrappers
//!
//! - **[`FilteredToolset`]**: Filter tools by predicate
//! - **[`PrefixedToolset`]**: Add name prefixes
//! - **[`RenamedToolset`]**: Rename specific tools
//! - **[`PreparedToolset`]**: Runtime tool modification
//! - **[`ApprovalRequiredToolset`]**: Require approval
//! - **[`WrapperToolset`]**: Pre/post processing hooks
//! - **[`ExternalToolset`]**: External tool execution
//!
//! ## Example
//!
//! ```rust
//! use serdes_ai_toolsets::{FunctionToolset, CombinedToolset, PrefixedToolset, AbstractToolset};
//! use serdes_ai_tools::{Tool, ToolDefinition, RunContext, ToolReturn, ToolError};
//! use async_trait::async_trait;
//!
//! struct SearchTool;
//!
//! #[async_trait]
//! impl Tool for SearchTool {
//!     fn definition(&self) -> ToolDefinition {
//!         ToolDefinition::new("search", "Search for items")
//!     }
//!
//!     async fn call(&self, _ctx: &RunContext, _args: serde_json::Value) -> Result<ToolReturn, ToolError> {
//!         Ok(ToolReturn::text("results"))
//!     }
//! }
//!
//! // Create toolsets
//! let web_tools = FunctionToolset::new().with_id("web").tool(SearchTool);
//! let local_tools = FunctionToolset::new().with_id("local").tool(SearchTool);
//!
//! // Prefix to avoid conflicts
//! let prefixed_web = PrefixedToolset::new(web_tools, "web");
//! let prefixed_local = PrefixedToolset::new(local_tools, "local");
//!
//! // Combine into one
//! let all_tools = CombinedToolset::new()
//!     .with_toolset(prefixed_web)
//!     .with_toolset(prefixed_local);
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod abstract_toolset;
pub mod approval;
pub mod combined;
pub mod dynamic;
pub mod external;
pub mod filtered;
pub mod function;
pub mod prefixed;
pub mod prepared;
pub mod renamed;
pub mod wrapper;

// Re-exports
pub use abstract_toolset::{
    AbstractToolset, BoxedToolset, ToolsetInfo, ToolsetResult, ToolsetTool,
};
pub use approval::{checkers as approval_checkers, ApprovalRequiredToolset};
pub use combined::CombinedToolset;
pub use dynamic::DynamicToolset;
pub use external::ExternalToolset;
pub use filtered::{filters, FilteredToolset};
pub use function::{AsyncFnTool, FunctionToolset};
pub use prefixed::PrefixedToolset;
pub use prepared::{preparers, PreparedToolset};
pub use renamed::RenamedToolset;
pub use wrapper::{LoggingWrapper, WrapperToolset};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        AbstractToolset, ApprovalRequiredToolset, BoxedToolset, CombinedToolset, DynamicToolset,
        ExternalToolset, FilteredToolset, FunctionToolset, PrefixedToolset, PreparedToolset,
        RenamedToolset, ToolsetInfo, ToolsetResult, ToolsetTool, WrapperToolset,
    };
}
