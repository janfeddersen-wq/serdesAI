//! MCP protocol types.
//!
//! This module defines the core types for the Model Context Protocol.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ============================================================================
// JSON-RPC Types
// ============================================================================

/// JSON-RPC request ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// Numeric ID.
    Number(i64),
    /// String ID.
    String(String),
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        Self::Number(n)
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

/// JSON-RPC request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (always "2.0").
    pub jsonrpc: String,
    /// Request ID.
    pub id: RequestId,
    /// Method name.
    pub method: String,
    /// Parameters (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<JsonValue>,
}

impl JsonRpcRequest {
    /// Create a new request.
    pub fn new(id: impl Into<RequestId>, method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            method: method.into(),
            params: None,
        }
    }

    /// Set parameters.
    pub fn with_params<T: Serialize>(mut self, params: T) -> Self {
        self.params = Some(serde_json::to_value(params).unwrap_or(JsonValue::Null));
        self
    }
}

/// JSON-RPC notification (no ID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Method name.
    pub method: String,
    /// Parameters (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<JsonValue>,
}

impl JsonRpcNotification {
    /// Create a new notification.
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params: None,
        }
    }

    /// Set parameters.
    pub fn with_params<T: Serialize>(mut self, params: T) -> Self {
        self.params = Some(serde_json::to_value(params).unwrap_or(JsonValue::Null));
        self
    }
}

/// JSON-RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Request ID.
    pub id: RequestId,
    /// Result (on success).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<JsonValue>,
    /// Error (on failure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: impl Into<RequestId>, result: impl Serialize) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            result: Some(serde_json::to_value(result).unwrap_or(JsonValue::Null)),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(id: impl Into<RequestId>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: id.into(),
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Check if this is an error response.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get the result, if present.
    pub fn into_result<T: for<'de> Deserialize<'de>>(self) -> Result<T, JsonRpcError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        match self.result {
            Some(v) => serde_json::from_value(v).map_err(|e| JsonRpcError {
                code: -32600,
                message: e.to_string(),
                data: None,
            }),
            None => Err(JsonRpcError {
                code: -32600,
                message: "Missing result".to_string(),
                data: None,
            }),
        }
    }
}

/// JSON-RPC error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i32,
    /// Error message.
    pub message: String,
    /// Additional data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for JsonRpcError {}

// Standard JSON-RPC error codes
impl JsonRpcError {
    /// Parse error (-32700).
    pub const PARSE_ERROR: i32 = -32700;
    /// Invalid request (-32600).
    pub const INVALID_REQUEST: i32 = -32600;
    /// Method not found (-32601).
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid params (-32602).
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal error (-32603).
    pub const INTERNAL_ERROR: i32 = -32603;
}

// ============================================================================
// MCP Types - Initialize
// ============================================================================

/// Implementation info (client or server).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    /// Implementation name.
    pub name: String,
    /// Implementation version.
    pub version: String,
}

impl Implementation {
    /// Create new implementation info.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Initialize request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version.
    pub protocol_version: String,
    /// Client capabilities.
    pub capabilities: ClientCapabilities,
    /// Client info.
    pub client_info: Implementation,
}

impl InitializeParams {
    /// Create with default capabilities.
    pub fn new(client_info: Implementation) -> Self {
        Self {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ClientCapabilities::default(),
            client_info,
        }
    }
}

/// Client capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Roots capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    /// Sampling capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
}

/// Roots capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootsCapability {
    /// Whether the client supports list changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Sampling capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Initialize result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol version.
    pub protocol_version: String,
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Server info.
    pub server_info: Implementation,
    /// Optional instructions for the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Server capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tools capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Resources capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Prompts capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Logging capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

/// Tools capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// Whether the server supports list changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Resources capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    /// Whether the server supports subscriptions.
    #[serde(default)]
    pub subscribe: bool,
    /// Whether the server supports list changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Prompts capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    /// Whether the server supports list changed notifications.
    #[serde(default)]
    pub list_changed: bool,
}

/// Logging capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingCapability {}

// ============================================================================
// MCP Types - Tools
// ============================================================================

/// MCP tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Input schema (JSON Schema).
    pub input_schema: JsonValue,
}

impl McpTool {
    /// Create a new tool.
    pub fn new(name: impl Into<String>, input_schema: JsonValue) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
        }
    }

    /// Set description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// List tools result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    /// Available tools.
    pub tools: Vec<McpTool>,
    /// Pagination cursor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Call tool parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    /// Tool name.
    pub name: String,
    /// Tool arguments.
    #[serde(default)]
    pub arguments: JsonValue,
}

impl CallToolParams {
    /// Create new call tool params.
    pub fn new(name: impl Into<String>, arguments: JsonValue) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

/// Call tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    /// Result content.
    pub content: Vec<ToolResultContent>,
    /// Whether this is an error result.
    #[serde(default)]
    pub is_error: bool,
}

impl CallToolResult {
    /// Create a success result with text.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text {
                text: text.into(),
            }],
            is_error: false,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text {
                text: message.into(),
            }],
            is_error: true,
        }
    }
}

/// Tool result content types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ToolResultContent {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// Text value.
        text: String,
    },
    /// Image content.
    #[serde(rename = "image")]
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type.
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Resource reference.
    #[serde(rename = "resource")]
    Resource {
        /// The resource.
        resource: ResourceContent,
    },
}

// ============================================================================
// MCP Types - Resources
// ============================================================================

/// Resource template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceTemplate {
    /// URI template.
    pub uri_template: String,
    /// Resource name.
    pub name: String,
    /// Resource description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Resource content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContent {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Binary content (base64).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

impl ResourceContent {
    /// Create text resource content.
    pub fn text(uri: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some("text/plain".to_string()),
            text: Some(text.into()),
            blob: None,
        }
    }

    /// Create binary resource content.
    pub fn binary(uri: impl Into<String>, data: &[u8], mime_type: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some(mime_type.into()),
            text: None,
            blob: Some(base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                data,
            )),
        }
    }
}

/// List resources result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListResourcesResult {
    /// Available resources.
    pub resources: Vec<ResourceTemplate>,
    /// Pagination cursor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Read resource parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    /// Resource URI.
    pub uri: String,
}

/// Read resource result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    /// Resource contents.
    pub contents: Vec<ResourceContent>,
}

// ============================================================================
// MCP Types - Prompts
// ============================================================================

/// Prompt definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// Prompt name.
    pub name: String,
    /// Prompt description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Prompt arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// Prompt argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    /// Argument name.
    pub name: String,
    /// Argument description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the argument is required.
    #[serde(default)]
    pub required: bool,
}

/// List prompts result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListPromptsResult {
    /// Available prompts.
    pub prompts: Vec<Prompt>,
    /// Pagination cursor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Get prompt result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    /// Prompt description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Prompt messages.
    pub messages: Vec<PromptMessage>,
}

/// Prompt message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    /// Role (user, assistant).
    pub role: String,
    /// Content.
    pub content: PromptMessageContent,
}

/// Prompt message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PromptMessageContent {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// Text value.
        text: String,
    },
    /// Resource content.
    #[serde(rename = "resource")]
    Resource {
        /// Resource.
        resource: ResourceContent,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id() {
        let id: RequestId = 42.into();
        assert!(matches!(id, RequestId::Number(42)));

        let id: RequestId = "test".into();
        assert!(matches!(id, RequestId::String(s) if s == "test"));
    }

    #[test]
    fn test_json_rpc_request() {
        let req = JsonRpcRequest::new(1, "test").with_params(serde_json::json!({"key": "value"}));

        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "test");
        assert!(req.params.is_some());
    }

    #[test]
    fn test_json_rpc_response() {
        let resp = JsonRpcResponse::success(1, "result");
        assert!(!resp.is_error());

        let resp = JsonRpcResponse::error(1, -32600, "Invalid");
        assert!(resp.is_error());
    }

    #[test]
    fn test_initialize_params() {
        let params = InitializeParams::new(Implementation::new("test", "1.0.0"));

        assert_eq!(params.protocol_version, "2024-11-05");
        assert_eq!(params.client_info.name, "test");
    }

    #[test]
    fn test_mcp_tool() {
        let tool = McpTool::new(
            "search",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        )
        .with_description("Search for things");

        assert_eq!(tool.name, "search");
        assert_eq!(tool.description, Some("Search for things".to_string()));
    }

    #[test]
    fn test_call_tool_result() {
        let result = CallToolResult::text("Hello!");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);

        let result = CallToolResult::error("Oops");
        assert!(result.is_error);
    }

    #[test]
    fn test_resource_content() {
        let content = ResourceContent::text("file:///test.txt", "Hello");
        assert_eq!(content.text, Some("Hello".to_string()));
        assert_eq!(content.mime_type, Some("text/plain".to_string()));
    }

    #[test]
    fn test_serialize_deserialize() {
        let tool = McpTool::new("test", serde_json::json!({"type": "object"}));
        let json = serde_json::to_string(&tool).unwrap();
        let parsed: McpTool = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test");
    }
}
