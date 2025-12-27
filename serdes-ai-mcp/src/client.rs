//! MCP client implementation.
//!
//! This module provides the `McpClient` for connecting to MCP servers.

use crate::error::{McpError, McpResult};
use crate::transport::{McpTransport, StdioTransport};
use crate::types::{
    CallToolParams, CallToolResult, Implementation, InitializeParams, InitializeResult,
    JsonRpcNotification, JsonRpcRequest, ListPromptsResult, ListResourcesResult,
    ListToolsResult, McpTool, ReadResourceParams, ReadResourceResult, RequestId,
    ServerCapabilities,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// MCP client for connecting to servers.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_mcp::McpClient;
///
/// let client = McpClient::stdio("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
/// client.initialize().await?;
///
/// let tools = client.list_tools().await?;
/// println!("Available tools: {:?}", tools);
/// ```
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    request_id: AtomicI64,
    server_capabilities: Mutex<Option<ServerCapabilities>>,
    server_info: Mutex<Option<Implementation>>,
    initialized: Mutex<bool>,
}

impl McpClient {
    /// Create a new client with a transport.
    pub fn new(transport: impl McpTransport + 'static) -> Self {
        Self {
            transport: Arc::new(transport),
            request_id: AtomicI64::new(1),
            server_capabilities: Mutex::new(None),
            server_info: Mutex::new(None),
            initialized: Mutex::new(false),
        }
    }

    /// Create a client that connects via stdio.
    pub async fn stdio(command: &str, args: &[&str]) -> McpResult<Self> {
        let transport = StdioTransport::spawn(command, args).await?;
        Ok(Self::new(transport))
    }

    /// Create a client with HTTP transport.
    #[cfg(feature = "reqwest")]
    pub fn http(url: &str) -> Self {
        use crate::transport::HttpTransport;
        Self::new(HttpTransport::new(url))
    }

    /// Initialize the connection.
    ///
    /// This must be called before using any other methods.
    pub async fn initialize(&self) -> McpResult<InitializeResult> {
        let params = InitializeParams::new(Implementation::new(
            "serdes-ai",
            env!("CARGO_PKG_VERSION"),
        ));

        let result: InitializeResult = self.call("initialize", params).await?;

        // Store server info
        *self.server_capabilities.lock().await = Some(result.capabilities.clone());
        *self.server_info.lock().await = Some(result.server_info.clone());

        // Send initialized notification
        self.notify("notifications/initialized", serde_json::Value::Null)
            .await?;
        *self.initialized.lock().await = true;

        Ok(result)
    }

    /// Check if the client is initialized.
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.lock().await
    }

    /// Get server capabilities.
    pub async fn server_capabilities(&self) -> Option<ServerCapabilities> {
        self.server_capabilities.lock().await.clone()
    }

    /// Get server info.
    pub async fn server_info(&self) -> Option<Implementation> {
        self.server_info.lock().await.clone()
    }

    // ========================================================================
    // Tools
    // ========================================================================

    /// List available tools.
    pub async fn list_tools(&self) -> McpResult<Vec<McpTool>> {
        self.ensure_initialized().await?;

        let result: ListToolsResult = self
            .call("tools/list", serde_json::Value::Null)
            .await?;
        Ok(result.tools)
    }

    /// Call a tool.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> McpResult<CallToolResult> {
        self.ensure_initialized().await?;

        let params = CallToolParams::new(name, arguments);
        self.call("tools/call", params).await
    }

    // ========================================================================
    // Resources
    // ========================================================================

    /// List available resources.
    pub async fn list_resources(&self) -> McpResult<ListResourcesResult> {
        self.ensure_initialized().await?;
        self.call("resources/list", serde_json::Value::Null).await
    }

    /// Read a resource.
    pub async fn read_resource(&self, uri: &str) -> McpResult<ReadResourceResult> {
        self.ensure_initialized().await?;

        let params = ReadResourceParams {
            uri: uri.to_string(),
        };
        self.call("resources/read", params).await
    }

    // ========================================================================
    // Prompts
    // ========================================================================

    /// List available prompts.
    pub async fn list_prompts(&self) -> McpResult<ListPromptsResult> {
        self.ensure_initialized().await?;
        self.call("prompts/list", serde_json::Value::Null).await
    }

    // ========================================================================
    // Connection
    // ========================================================================

    /// Close the connection.
    pub async fn close(&self) -> McpResult<()> {
        *self.initialized.lock().await = false;
        self.transport.close().await
    }

    /// Check if connected.
    pub fn is_connected(&self) -> bool {
        self.transport.is_connected()
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    async fn ensure_initialized(&self) -> McpResult<()> {
        if !*self.initialized.lock().await {
            return Err(McpError::NotInitialized);
        }
        Ok(())
    }

    fn next_id(&self) -> RequestId {
        RequestId::Number(self.request_id.fetch_add(1, Ordering::SeqCst))
    }

    async fn call<P: Serialize, R: DeserializeOwned>(
        &self,
        method: &str,
        params: P,
    ) -> McpResult<R> {
        let request = JsonRpcRequest::new(self.next_id(), method).with_params(params);

        let response = self.transport.request(&request).await?;

        if let Some(error) = response.error {
            return Err(McpError::Protocol {
                code: error.code,
                message: error.message,
            });
        }

        let result = response.result.ok_or(McpError::NoResult)?;
        serde_json::from_value(result).map_err(McpError::from)
    }

    async fn notify<P: Serialize>(&self, method: &str, params: P) -> McpResult<()> {
        let notification = JsonRpcNotification::new(method).with_params(params);
        self.transport.notify(&notification).await
    }
}

/// Builder for creating MCP clients.
pub struct McpClientBuilder {
    command: Option<String>,
    args: Vec<String>,
    url: Option<String>,
}

impl Default for McpClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl McpClientBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            command: None,
            args: Vec::new(),
            url: None,
        }
    }

    /// Set the command for stdio transport.
    pub fn command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }

    /// Add an argument for stdio transport.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set multiple arguments for stdio transport.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(|s| s.into()));
        self
    }

    /// Set the URL for HTTP transport.
    #[cfg(feature = "reqwest")]
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Build the client.
    pub async fn build(self) -> McpResult<McpClient> {
        #[cfg(feature = "reqwest")]
        if let Some(url) = self.url {
            return Ok(McpClient::http(&url));
        }

        if let Some(command) = self.command {
            let args: Vec<&str> = self.args.iter().map(|s| s.as_str()).collect();
            return McpClient::stdio(&command, &args).await;
        }

        Err(McpError::Other(
            "No transport configured".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MemoryTransport;
    use crate::types::JsonRpcResponse;

    #[tokio::test]
    async fn test_client_not_initialized() {
        let transport = MemoryTransport::new();
        let client = McpClient::new(transport);

        let result = client.list_tools().await;
        assert!(matches!(result, Err(McpError::NotInitialized)));
    }

    #[tokio::test]
    async fn test_client_initialize() {
        let transport = MemoryTransport::new();

        // Push initialize response
        let init_result = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ServerCapabilities::default(),
            server_info: Implementation::new("test-server", "1.0.0"),
            instructions: None,
        };
        transport
            .push_response(JsonRpcResponse::success(1, init_result))
            .await;

        let client = McpClient::new(transport);
        let result = client.initialize().await.unwrap();

        assert_eq!(result.server_info.name, "test-server");
        assert!(client.is_initialized().await);
    }

    #[tokio::test]
    async fn test_client_builder() {
        let builder = McpClientBuilder::new()
            .command("echo")
            .arg("test")
            .args(["arg1", "arg2"]);

        // We can't actually build without a real command, but we can check the config
        assert!(builder.command.is_some());
        assert_eq!(builder.args.len(), 3);
    }

    #[tokio::test]
    async fn test_next_id() {
        let transport = MemoryTransport::new();
        let client = McpClient::new(transport);

        let id1 = client.next_id();
        let id2 = client.next_id();

        // IDs should be incrementing
        if let (RequestId::Number(n1), RequestId::Number(n2)) = (id1, id2) {
            assert_eq!(n2, n1 + 1);
        } else {
            panic!("Expected numeric IDs");
        }
    }
}
