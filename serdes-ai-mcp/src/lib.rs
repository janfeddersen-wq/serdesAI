//! # serdes-ai-mcp
//!
//! Model Context Protocol (MCP) implementation for serdes-ai.
//!
//! MCP is a protocol for sharing context between LLM applications.
//! This crate provides both client and server implementations.
//!
//! ## Core Concepts
//!
//! - **[`McpClient`]**: Connect to MCP servers and access their tools
//! - **[`McpServer`]**: Expose tools via MCP protocol
//! - **[`McpTransport`]**: Transport layer abstraction (stdio, HTTP)
//! - **[`McpToolset`]**: Automatically import tools from MCP servers
//!
//! ## Feature Flags
//!
//! - `client` (default): MCP client implementation
//! - `server`: MCP server implementation
//! - `full`: Both client and server
//!
//! ## Example - Client
//!
//! ```ignore
//! use serdes_ai_mcp::{McpClient, McpToolset};
//!
//! // Connect to an MCP server
//! let client = McpClient::stdio("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
//! client.initialize().await?;
//!
//! // List available tools
//! let tools = client.list_tools().await?;
//! for tool in tools {
//!     println!("Tool: {} - {}", tool.name, tool.description.unwrap_or_default());
//! }
//!
//! // Call a tool
//! let result = client.call_tool("read_file", serde_json::json!({
//!     "path": "/tmp/test.txt"
//! })).await?;
//! ```
//!
//! ## Example - Server
//!
//! ```ignore
//! use serdes_ai_mcp::{McpServer, McpTool, CallToolResult};
//!
//! let server = McpServer::new("my-server", "1.0.0")
//!     .tool_fn(
//!         McpTool::new("echo", serde_json::json!({"type": "object"}))
//!             .with_description("Echo the input"),
//!         |args| CallToolResult::text(args.to_string()),
//!     );
//!
//! server.run_stdio().await?;
//! ```
//!
//! ## Example - Toolset Integration
//!
//! ```ignore
//! use serdes_ai_mcp::McpToolset;
//! use serdes_ai_agent::agent;
//!
//! // Create toolset from MCP server
//! let toolset = McpToolset::stdio("npx", &["-y", "@mcp/server-fs"]).await?
//!     .with_id("filesystem");
//!
//! // Use with agent
//! let agent = agent(model)
//!     .toolset(toolset)
//!     .build();
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod error;
pub mod resources;
pub mod toolset;
pub mod transport;
pub mod types;

#[cfg(feature = "client")]
#[cfg_attr(docsrs, doc(cfg(feature = "client")))]
pub mod client;

#[cfg(feature = "server")]
#[cfg_attr(docsrs, doc(cfg(feature = "server")))]
pub mod server;

// Re-exports
pub use error::{McpError, McpResult};
pub use resources::{parse_resource_uri, read_file_resource, ResourceManager, ResourceUri};
pub use toolset::{load_mcp_servers, McpServerConfig, McpToolset, McpTransportConfig};
pub use transport::{McpTransport, MemoryTransport, StdioTransport};
pub use types::{
    CallToolParams, CallToolResult, ClientCapabilities, Implementation, InitializeParams,
    InitializeResult, JsonRpcError, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    ListPromptsResult, ListResourcesResult, ListToolsResult, McpTool, Prompt, PromptArgument,
    ReadResourceParams, ReadResourceResult, RequestId, ResourceContent, ResourceTemplate,
    ServerCapabilities, ToolResultContent,
};

#[cfg(feature = "client")]
pub use client::{McpClient, McpClientBuilder};

#[cfg(feature = "server")]
pub use server::{AsyncFnToolHandler, FnToolHandler, McpServer, ToolHandler};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{CallToolResult, McpError, McpResult, McpTool, McpToolset, McpTransport};

    #[cfg(feature = "client")]
    pub use crate::{McpClient, McpClientBuilder};

    #[cfg(feature = "server")]
    pub use crate::{McpServer, ToolHandler};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let tool = McpTool::new("test", serde_json::json!({"type": "object"}));
        assert_eq!(tool.name, "test");
    }
}
