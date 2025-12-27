//! # serdes-ai-tools
//!
//! Tool system for serdes-ai agents.
//!
//! This crate provides the infrastructure for defining, registering, and
//! executing tools that agents can use during conversations.
//!
//! ## Core Concepts
//!
//! - **[`Tool`]**: Trait for callable tools with typed parameters
//! - **[`ToolRegistry`]**: Manage and lookup registered tools
//! - **[`ToolDefinition`]**: JSON Schema-based tool descriptions for LLMs
//! - **[`RunContext`]**: Execution context with dependencies passed to tools
//! - **[`ToolReturn`]**: Return values from tool execution
//!
//! ## Defining Tools
//!
//! Tools can be defined by implementing the [`Tool`] trait:
//!
//! ```rust
//! use async_trait::async_trait;
//! use serdes_ai_tools::{
//!     Tool, ToolDefinition, ObjectJsonSchema, PropertySchema,
//!     RunContext, ToolResult, ToolReturn,
//! };
//!
//! struct WeatherTool;
//!
//! #[async_trait]
//! impl Tool for WeatherTool {
//!     fn definition(&self) -> ToolDefinition {
//!         ToolDefinition::new("get_weather", "Get current weather for a location")
//!             .with_parameters(
//!                 ObjectJsonSchema::new()
//!                     .with_property(
//!                         "location",
//!                         PropertySchema::string("City name").build(),
//!                         true,
//!                     )
//!             )
//!     }
//!
//!     async fn call(
//!         &self,
//!         _ctx: &RunContext,
//!         args: serde_json::Value,
//!     ) -> ToolResult {
//!         let location = args["location"].as_str().unwrap_or("Unknown");
//!         Ok(ToolReturn::text(format!("Weather in {}: 72°F, sunny", location)))
//!     }
//! }
//! ```
//!
//! ## Using the Registry
//!
//! ```rust
//! use serdes_ai_tools::{ToolRegistry, Tool, RunContext};
//!
//! # struct WeatherTool;
//! # use async_trait::async_trait;
//! # use serdes_ai_tools::{ToolDefinition, ObjectJsonSchema, ToolResult, ToolReturn};
//! # #[async_trait] impl Tool for WeatherTool {
//! #     fn definition(&self) -> ToolDefinition { ToolDefinition::new("weather", "Get weather") }
//! #     async fn call(&self, _: &RunContext, _: serde_json::Value) -> ToolResult { Ok(ToolReturn::empty()) }
//! # }
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(WeatherTool);
//!
//! // Get all definitions for the model
//! let definitions = registry.definitions();
//!
//! // Check if a tool exists
//! assert!(registry.contains("weather"));
//! ```
//!
//! ## Builtin Tools
//!
//! The crate provides builtin tools for common operations:
//!
//! - [`builtin::WebSearchTool`]: Search the web for information
//! - [`builtin::CodeExecutionTool`]: Execute code in a sandbox
//! - [`builtin::FileSearchTool`]: Vector-based file search
//!
//! ## Common Tools (Third-Party Integrations)
//!
//! With the `common-tools` feature, additional third-party tool integrations are available:
//!
//! ```toml
//! serdes-ai-tools = { version = "0.1", features = ["common-tools"] }
//! ```
//!
//! - `common::DuckDuckGoTool`: Web search using DuckDuckGo (no API key required)
//! - `common::TavilyTool`: AI-optimized search using Tavily's API

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod builtin;
#[cfg(feature = "common-tools")]
pub mod common;
pub mod context;
pub mod deferred;
pub mod definition;
pub mod errors;
pub mod registry;
pub mod return_types;
pub mod schema;
pub mod tool;

// Re-export core types
pub use context::RunContext;
pub use deferred::{
    DeferredToolCall, DeferredToolDecision, DeferredToolDecisions, DeferredToolRequests,
    DeferredToolResult, DeferredToolResults, ToolApproved, ToolApprover, ToolDenied,
};
pub use definition::{ObjectJsonSchema, PropertySchema, ToolDefinition};
pub use errors::{ToolError, ToolErrorInfo};
pub use registry::{ToolProvider, ToolRegistry};
pub use return_types::{IntoToolReturn, SerializableToolResult, ToolResult, ToolReturn};
pub use schema::{types as schema_types, JsonSchemaGenerator, SchemaBuilder};
pub use tool::{BoxedTool, FunctionTool, SyncFunctionTool, Tool};

// Delete old files if they exist
#[allow(unused_imports)]
mod _cleanup {
    // This module exists only to document what was removed
    // Old files: error.rs, result.rs were replaced with errors.rs, return_types.rs
}
