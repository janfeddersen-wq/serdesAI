//! MCP toolset for integrating MCP servers with agents.
//!
//! This module provides `McpToolset` which implements `AbstractToolset`
//! to automatically import tools from MCP servers.

use crate::client::McpClient;
use crate::error::{McpError, McpResult};
use crate::types::{McpTool, ToolResultContent};
use async_trait::async_trait;
use parking_lot::RwLock;
use serdes_ai_tools::definition::ToolDefinition;
use serdes_ai_tools::return_types::ToolReturn;
use serdes_ai_tools::ToolError;
use serdes_ai_toolsets::abstract_toolset::{AbstractToolset, ToolsetTool};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Toolset that wraps an MCP server's tools.
///
/// This toolset automatically fetches and exposes tools from an MCP server.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_mcp::{McpClient, McpToolset};
///
/// // Connect to MCP server
/// let client = McpClient::stdio("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
/// client.initialize().await?;
///
/// // Create toolset
/// let toolset = McpToolset::new(client).with_id("filesystem");
///
/// // Use with agent
/// let agent = agent(model)
///     .toolset(toolset)
///     .build();
/// ```
pub struct McpToolset<Deps = ()> {
    id: Option<String>,
    client: Arc<Mutex<McpClient>>,
    tools_cache: RwLock<Option<Vec<McpTool>>>,
    _phantom: PhantomData<Deps>,
}

impl<Deps> McpToolset<Deps> {
    /// Create a new MCP toolset from a client.
    pub fn new(client: McpClient) -> Self {
        Self {
            id: None,
            client: Arc::new(Mutex::new(client)),
            tools_cache: RwLock::new(None),
            _phantom: PhantomData,
        }
    }

    /// Set the toolset ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Connect via stdio and create toolset.
    pub async fn stdio(command: &str, args: &[&str]) -> McpResult<Self> {
        let client = McpClient::stdio(command, args).await?;
        client.initialize().await?;
        Ok(Self::new(client))
    }

    /// Connect via HTTP and create toolset.
    #[cfg(feature = "reqwest")]
    pub async fn http(url: &str) -> McpResult<Self> {
        let client = McpClient::http(url);
        client.initialize().await?;
        Ok(Self::new(client))
    }

    /// Refresh the tools cache.
    pub async fn refresh(&self) -> McpResult<()> {
        let client = self.client.lock().await;
        let tools = client.list_tools().await?;
        *self.tools_cache.write() = Some(tools);
        Ok(())
    }

    /// Get cached tools.
    pub fn cached_tools(&self) -> Option<Vec<McpTool>> {
        self.tools_cache.read().clone()
    }

    fn convert_to_toolset_tool(
        &self,
        mcp_tool: &McpTool,
    ) -> ToolsetTool {
        let definition = ToolDefinition::new(
            mcp_tool.name.clone(),
            mcp_tool.description.clone().unwrap_or_default(),
        );

        ToolsetTool::new(definition)
            .with_max_retries(2)
    }

    fn convert_tools_to_map(
        &self,
        tools: &[McpTool],
    ) -> HashMap<String, ToolsetTool> {
        tools
            .iter()
            .map(|t| (t.name.clone(), self.convert_to_toolset_tool(t)))
            .collect()
    }
}

#[async_trait]
impl<Deps: Send + Sync + 'static> AbstractToolset<Deps> for McpToolset<Deps> {
    fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    async fn get_tools(
        &self,
        _ctx: &serdes_ai_tools::RunContext<Deps>,
    ) -> Result<HashMap<String, ToolsetTool>, ToolError> {
        // Check cache first
        {
            let cache = self.tools_cache.read();
            if let Some(ref tools) = *cache {
                return Ok(self.convert_tools_to_map(tools));
            }
        }

        // Fetch from server
        let client = self.client.lock().await;
        let tools = client.list_tools().await.map_err(|e| ToolError::ExecutionFailed {
            message: e.to_string(),
            retryable: false,
        })?;

        // Update cache
        *self.tools_cache.write() = Some(tools.clone());

        Ok(self.convert_tools_to_map(&tools))
    }

    async fn call_tool(
        &self,
        name: &str,
        args: JsonValue,
        _ctx: &serdes_ai_tools::RunContext<Deps>,
        _tool: &ToolsetTool,
    ) -> Result<ToolReturn, ToolError> {
        let client = self.client.lock().await;
        let result = client
            .call_tool(name, args)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                message: e.to_string(),
                retryable: matches!(e, McpError::Timeout | McpError::Transport(_)),
            })?;

        if result.is_error {
            return Err(ToolError::ExecutionFailed {
                message: result
                    .content
                    .first()
                    .map(|c| match c {
                        ToolResultContent::Text { text } => text.clone(),
                        _ => "Unknown error".to_string(),
                    })
                    .unwrap_or_else(|| "Unknown error".to_string()),
                retryable: false,
            });
        }

        // Convert result to ToolReturn
        let content = result
            .content
            .into_iter()
            .next()
            .map(|c| match c {
                ToolResultContent::Text { text } => ToolReturn::text(text),
                ToolResultContent::Image { data, mime_type } => {
                    // For now, return as JSON with base64 data
                    ToolReturn::json(serde_json::json!({
                        "type": "image",
                        "data": data,
                        "mimeType": mime_type
                    }))
                }
                ToolResultContent::Resource { resource } => {
                    ToolReturn::text(resource.text.unwrap_or_default())
                }
            })
            .unwrap_or_else(ToolReturn::empty);

        Ok(content)
    }

    async fn enter(&self) -> Result<(), ToolError> {
        // Ensure client is initialized
        let client = self.client.lock().await;
        if !client.is_initialized().await {
            client.initialize().await.map_err(|e| ToolError::ExecutionFailed {
                message: e.to_string(),
                retryable: false,
            })?;
        }
        Ok(())
    }

    async fn exit(&self) -> Result<(), ToolError> {
        let client = self.client.lock().await;
        client.close().await.map_err(|e| ToolError::ExecutionFailed {
            message: e.to_string(),
            retryable: false,
        })
    }
}

/// Configuration for an MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Server name/ID.
    pub name: String,
    /// Transport configuration.
    pub transport: McpTransportConfig,
}

impl McpServerConfig {
    /// Create a stdio server config.
    pub fn stdio(
        name: impl Into<String>,
        command: impl Into<String>,
        args: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            transport: McpTransportConfig::Stdio {
                command: command.into(),
                args,
            },
        }
    }

    /// Create an HTTP server config.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transport: McpTransportConfig::Http { url: url.into() },
        }
    }
}

/// Transport configuration.
#[derive(Debug, Clone)]
pub enum McpTransportConfig {
    /// Stdio transport (local process).
    Stdio {
        /// Command to run.
        command: String,
        /// Command arguments.
        args: Vec<String>,
    },
    /// HTTP transport (remote server).
    Http {
        /// Server URL.
        url: String,
    },
    /// SSE transport (remote server with streaming).
    Sse {
        /// Server URL.
        url: String,
    },
}

/// Load MCP servers from configuration.
pub async fn load_mcp_servers(
    configs: Vec<McpServerConfig>,
) -> McpResult<Vec<McpToolset<()>>> {
    let mut toolsets = Vec::new();

    for config in configs {
        let toolset = match config.transport {
            McpTransportConfig::Stdio { command, args } => {
                let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                McpToolset::stdio(&command, &args_refs).await?
            }
            McpTransportConfig::Http { url: _ } => {
                #[cfg(feature = "reqwest")]
                {
                    if let McpTransportConfig::Http { url } = &config.transport {
                        McpToolset::http(url).await?
                    } else {
                        unreachable!()
                    }
                }
                #[cfg(not(feature = "reqwest"))]
                {
                    return Err(McpError::Other(
                        "HTTP transport requires 'reqwest' feature".to_string(),
                    ));
                }
            }
            McpTransportConfig::Sse { url: _ } => {
                #[cfg(feature = "reqwest")]
                {
                    if let McpTransportConfig::Sse { url } = &config.transport {
                        McpToolset::http(url).await?
                    } else {
                        unreachable!()
                    }
                }
                #[cfg(not(feature = "reqwest"))]
                {
                    return Err(McpError::Other(
                        "SSE transport requires 'reqwest' feature".to_string(),
                    ));
                }
            }
        };
        toolsets.push(toolset.with_id(config.name));
    }

    Ok(toolsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config() {
        let config = McpServerConfig::stdio(
            "fs",
            "npx",
            vec!["-y".to_string(), "@mcp/server-fs".to_string()],
        );
        assert_eq!(config.name, "fs");
        assert!(matches!(config.transport, McpTransportConfig::Stdio { .. }));
    }

    #[test]
    fn test_http_config() {
        let config = McpServerConfig::http("remote", "http://localhost:8080");
        assert_eq!(config.name, "remote");
        assert!(matches!(config.transport, McpTransportConfig::Http { .. }));
    }

    #[test]
    fn test_convert_tool() {
        let mcp_tool = McpTool::new(
            "search",
            serde_json::json!({"type": "object"}),
        )
        .with_description("Search for things");

        // We can't easily test the full conversion without a client,
        // but we can check the tool is well-formed
        assert_eq!(mcp_tool.name, "search");
        assert_eq!(mcp_tool.description, Some("Search for things".to_string()));
    }
}
