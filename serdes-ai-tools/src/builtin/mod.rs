//! Builtin tools for common operations.
//!
//! This module provides ready-to-use tools for common agent operations:
//!
//! - **Web Search**: Search the web for information
//! - **Web Fetch**: Fetch content from URLs (Anthropic, Google)
//! - **Code Execution**: Execute code in a sandbox
//! - **File Search**: Vector-based file search
//! - **Image Generation**: Generate images from text prompts
//! - **Memory**: Agent memory (Anthropic-specific)
//! - **MCP Server**: Reference MCP servers at the API level (OpenAI, Anthropic)
//!
//! These tools are designed to be easily integrated with external services
//! while providing sensible defaults.

pub mod code_execution;
pub mod file_search;
pub mod image_gen;
pub mod mcp_server;
pub mod memory;
pub mod web_fetch;
pub mod web_search;

pub use code_execution::{CodeExecutionConfig, CodeExecutionTool, ProgrammingLanguage};
pub use file_search::{FileSearchConfig, FileSearchTool};
pub use image_gen::{
    ImageAspectRatio, ImageBackground, ImageGenerationTool, ImageQuality, ImageSize, OutputFormat,
};
pub use mcp_server::MCPServerTool;
pub use memory::MemoryTool;
pub use web_fetch::{WebFetchConfig, WebFetchError, WebFetchTool, WebFetchToolBuilder};
pub use web_search::{
    SearchContextSize, SearchDepth, UserLocation, WebSearchConfig, WebSearchError, WebSearchTool,
    WebSearchToolBuilder,
};
