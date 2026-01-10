//! MCP server implementation.
//!
//! This module provides types for building MCP servers.

use crate::error::{McpError, McpResult};
use crate::types::{
    CallToolParams, CallToolResult, Implementation, InitializeResult, JsonRpcMessage,
    JsonRpcResponse, ListToolsResult, McpTool, ServerCapabilities, ToolsCapability,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Trait for MCP tool handlers.
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Get the tool definition.
    fn definition(&self) -> McpTool;

    /// Handle a tool call.
    async fn call(&self, arguments: JsonValue) -> McpResult<CallToolResult>;
}

/// Simple function-based tool handler.
pub struct FnToolHandler<F>
where
    F: Fn(JsonValue) -> CallToolResult + Send + Sync,
{
    definition: McpTool,
    handler: F,
}

impl<F> FnToolHandler<F>
where
    F: Fn(JsonValue) -> CallToolResult + Send + Sync,
{
    /// Create a new function tool handler.
    pub fn new(definition: McpTool, handler: F) -> Self {
        Self {
            definition,
            handler,
        }
    }
}

#[async_trait]
impl<F> ToolHandler for FnToolHandler<F>
where
    F: Fn(JsonValue) -> CallToolResult + Send + Sync,
{
    fn definition(&self) -> McpTool {
        self.definition.clone()
    }

    async fn call(&self, arguments: JsonValue) -> McpResult<CallToolResult> {
        Ok((self.handler)(arguments))
    }
}

/// Async function-based tool handler.
pub struct AsyncFnToolHandler<F, Fut>
where
    F: Fn(JsonValue) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = CallToolResult> + Send,
{
    definition: McpTool,
    handler: F,
}

impl<F, Fut> AsyncFnToolHandler<F, Fut>
where
    F: Fn(JsonValue) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = CallToolResult> + Send,
{
    /// Create a new async function tool handler.
    pub fn new(definition: McpTool, handler: F) -> Self {
        Self {
            definition,
            handler,
        }
    }
}

#[async_trait]
impl<F, Fut> ToolHandler for AsyncFnToolHandler<F, Fut>
where
    F: Fn(JsonValue) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = CallToolResult> + Send,
{
    fn definition(&self) -> McpTool {
        self.definition.clone()
    }

    async fn call(&self, arguments: JsonValue) -> McpResult<CallToolResult> {
        Ok((self.handler)(arguments).await)
    }
}

/// MCP server for exposing tools.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_mcp::{McpServer, McpTool, CallToolResult};
///
/// let server = McpServer::new("my-server", "1.0.0")
///     .tool_fn(
///         McpTool::new("echo", serde_json::json!({"type": "object"}))
///             .with_description("Echo the input"),
///         |args| CallToolResult::text(args.to_string()),
///     );
///
/// server.run_stdio().await?;
/// ```
pub struct McpServer {
    info: Implementation,
    tools: RwLock<HashMap<String, Arc<dyn ToolHandler>>>,
    capabilities: ServerCapabilities,
}

impl McpServer {
    /// Create a new server.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            info: Implementation::new(name, version),
            tools: RwLock::new(HashMap::new()),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: false,
                }),
                ..Default::default()
            },
        }
    }

    /// Add a tool handler.
    pub fn tool(self, handler: impl ToolHandler + 'static) -> Self {
        let def = handler.definition();
        self.tools
            .write()
            .insert(def.name.clone(), Arc::new(handler));
        self
    }

    /// Add a sync function tool.
    pub fn tool_fn<F>(self, definition: McpTool, handler: F) -> Self
    where
        F: Fn(JsonValue) -> CallToolResult + Send + Sync + 'static,
    {
        let name = definition.name.clone();
        let handler = FnToolHandler::new(definition, handler);
        self.tools.write().insert(name, Arc::new(handler));
        self
    }

    /// Run the server on stdio.
    pub async fn run_stdio(&self) -> McpResult<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let response = self.handle_message(trimmed).await;

                    if let Some(resp) = response {
                        let json = serde_json::to_string(&resp).map_err(|e| McpError::Json(e))?;
                        stdout.write_all(json.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await?;
                    }
                }
                Err(e) => return Err(McpError::Io(e)),
            }
        }

        Ok(())
    }

    async fn handle_message(&self, message: &str) -> Option<JsonRpcResponse> {
        let message: JsonRpcMessage = match serde_json::from_str(message) {
            Ok(r) => r,
            Err(e) => {
                return Some(JsonRpcResponse::error(
                    0,
                    -32700,
                    format!("Parse error: {}", e),
                ));
            }
        };

        let request = match message {
            JsonRpcMessage::Notification(notification) => {
                if notification.method.starts_with("notifications/") {
                    return None;
                }
                return None;
            }
            JsonRpcMessage::Request(request) => request,
        };

        match request.method.as_str() {
            "initialize" => {
                let result = InitializeResult {
                    protocol_version: "2024-11-05".to_string(),
                    capabilities: self.capabilities.clone(),
                    server_info: self.info.clone(),
                    instructions: None,
                };
                Some(JsonRpcResponse::success(request.id, result))
            }
            "tools/list" => {
                let tools: Vec<McpTool> =
                    self.tools.read().values().map(|h| h.definition()).collect();
                let result = ListToolsResult {
                    tools,
                    next_cursor: None,
                };
                Some(JsonRpcResponse::success(request.id, result))
            }
            "tools/call" => {
                let params: CallToolParams = match request.params {
                    Some(p) => match serde_json::from_value(p) {
                        Ok(params) => params,
                        Err(e) => {
                            return Some(JsonRpcResponse::error(
                                request.id,
                                -32602,
                                format!("Invalid params: {}", e),
                            ));
                        }
                    },
                    None => {
                        return Some(JsonRpcResponse::error(request.id, -32602, "Missing params"));
                    }
                };

                let handler = match self.tools.read().get(&params.name) {
                    Some(h) => h.clone(),
                    None => {
                        return Some(JsonRpcResponse::error(
                            request.id,
                            -32602,
                            format!("Tool not found: {}", params.name),
                        ));
                    }
                };

                let result = match handler.call(params.arguments).await {
                    Ok(output) => output,
                    Err(e) => CallToolResult::error(e.to_string()),
                };
                Some(JsonRpcResponse::success(request.id, result))
            }
            _ => Some(JsonRpcResponse::error(
                request.id,
                -32601,
                format!("Method not found: {}", request.method),
            )),
        }
    }

    /// Get server info.
    pub fn info(&self) -> &Implementation {
        &self.info
    }

    /// Get registered tool count.
    pub fn tool_count(&self) -> usize {
        self.tools.read().len()
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new("mcp-server", "0.1.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = McpServer::new("test", "1.0.0");
        assert_eq!(server.info().name, "test");
        assert_eq!(server.tool_count(), 0);
    }

    #[test]
    fn test_add_tool() {
        let tool = McpTool::new("echo", serde_json::json!({"type": "object"}))
            .with_description("Echo input");

        let server = McpServer::new("test", "1.0.0")
            .tool_fn(tool, |args| CallToolResult::text(args.to_string()));

        assert_eq!(server.tool_count(), 1);
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        let server = McpServer::new("test", "1.0.0");

        let message = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let response = server.handle_message(message).await.unwrap();

        assert!(!response.is_error());
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let tool = McpTool::new("test", serde_json::json!({"type": "object"}));
        let server = McpServer::new("test", "1.0.0").tool_fn(tool, |_| CallToolResult::text("ok"));

        let message = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let response = server.handle_message(message).await.unwrap();

        assert!(!response.is_error());
        let result: ListToolsResult = serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.tools.len(), 1);
    }

    #[tokio::test]
    async fn test_handle_tools_call() {
        let tool = McpTool::new("echo", serde_json::json!({"type": "object"}));
        let server = McpServer::new("test", "1.0.0")
            .tool_fn(tool, |args| CallToolResult::text(args.to_string()));

        let message = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hello"}}}"#;
        let response = server.handle_message(message).await.unwrap();

        assert!(!response.is_error());
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let server = McpServer::new("test", "1.0.0");

        let message = r#"{"jsonrpc":"2.0","id":1,"method":"unknown"}"#;
        let response = server.handle_message(message).await.unwrap();

        assert!(response.is_error());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_handle_notification() {
        let server = McpServer::new("test", "1.0.0");

        let message = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let response = server.handle_message(message).await;

        // Notifications don't get responses
        assert!(response.is_none());
    }
}
