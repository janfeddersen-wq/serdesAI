//! MCP Server tool for referencing MCP servers at the API level.
//!
//! This module provides a builtin tool type for referencing Model Context Protocol
//! (MCP) servers. Supported by OpenAI Responses API and Anthropic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Builtin tool for referencing MCP servers at the API level.
///
/// This tool allows agents to interact with external MCP servers for
/// enhanced capabilities. Supported by OpenAI Responses and Anthropic.
///
/// # Example
///
/// ```
/// use serdes_ai_tools::builtin::MCPServerTool;
///
/// let server = MCPServerTool::new("my-server", "https://mcp.example.com")
///     .with_description("My awesome MCP server")
///     .with_auth("secret-token")
///     .with_allowed_tools(vec!["tool1".to_string(), "tool2".to_string()]);
///
/// assert_eq!(server.unique_id(), "mcp_server:my-server");
/// assert_eq!(server.label(), "MCP: my-server");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MCPServerTool {
    /// Unique identifier for this MCP server.
    pub id: String,
    /// URL of the MCP server.
    pub url: String,
    /// Optional authorization token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization_token: Option<String>,
    /// Optional description of the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Allowed tools from this server (if None, all tools are allowed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// Optional HTTP headers for requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// The kind identifier for this tool type.
    pub kind: String,
}

impl MCPServerTool {
    /// The static kind identifier for MCP server tools.
    pub const KIND: &'static str = "mcp_server";

    /// Create a new MCP server tool with required fields.
    #[must_use]
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            authorization_token: None,
            description: None,
            allowed_tools: None,
            headers: None,
            kind: Self::KIND.into(),
        }
    }

    /// Get the kind identifier.
    #[must_use]
    pub fn kind() -> &'static str {
        Self::KIND
    }

    /// Set the authorization token.
    #[must_use]
    pub fn with_auth(mut self, token: impl Into<String>) -> Self {
        self.authorization_token = Some(token.into());
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the allowed tools.
    #[must_use]
    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = Some(tools);
        self
    }

    /// Add a single allowed tool.
    #[must_use]
    pub fn allow_tool(mut self, tool: impl Into<String>) -> Self {
        match &mut self.allowed_tools {
            Some(tools) => tools.push(tool.into()),
            None => self.allowed_tools = Some(vec![tool.into()]),
        }
        self
    }

    /// Set HTTP headers for requests.
    #[must_use]
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Add a single HTTP header.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        match &mut self.headers {
            Some(headers) => {
                headers.insert(key.into(), value.into());
            }
            None => {
                let mut headers = HashMap::new();
                headers.insert(key.into(), value.into());
                self.headers = Some(headers);
            }
        }
        self
    }

    /// Get a unique identifier combining kind and id.
    #[must_use]
    pub fn unique_id(&self) -> String {
        format!("{}:{}", self.kind, self.id)
    }

    /// Get a human-readable label for this server.
    #[must_use]
    pub fn label(&self) -> String {
        format!("MCP: {}", self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_server_new() {
        let server = MCPServerTool::new("test-server", "https://example.com/mcp");
        assert_eq!(server.id, "test-server");
        assert_eq!(server.url, "https://example.com/mcp");
        assert_eq!(server.kind, "mcp_server");
        assert!(server.authorization_token.is_none());
        assert!(server.description.is_none());
        assert!(server.allowed_tools.is_none());
        assert!(server.headers.is_none());
    }

    #[test]
    fn test_mcp_server_kind() {
        assert_eq!(MCPServerTool::kind(), "mcp_server");
    }

    #[test]
    fn test_mcp_server_with_auth() {
        let server = MCPServerTool::new("test", "https://example.com")
            .with_auth("my-token");
        assert_eq!(server.authorization_token, Some("my-token".to_string()));
    }

    #[test]
    fn test_mcp_server_with_description() {
        let server = MCPServerTool::new("test", "https://example.com")
            .with_description("A test server");
        assert_eq!(server.description, Some("A test server".to_string()));
    }

    #[test]
    fn test_mcp_server_with_allowed_tools() {
        let server = MCPServerTool::new("test", "https://example.com")
            .with_allowed_tools(vec!["tool1".to_string(), "tool2".to_string()]);
        assert_eq!(server.allowed_tools, Some(vec!["tool1".to_string(), "tool2".to_string()]));
    }

    #[test]
    fn test_mcp_server_allow_tool() {
        let server = MCPServerTool::new("test", "https://example.com")
            .allow_tool("tool1")
            .allow_tool("tool2");
        assert_eq!(server.allowed_tools, Some(vec!["tool1".to_string(), "tool2".to_string()]));
    }

    #[test]
    fn test_mcp_server_with_headers() {
        let mut headers = HashMap::new();
        headers.insert("X-Custom".to_string(), "value".to_string());
        
        let server = MCPServerTool::new("test", "https://example.com")
            .with_headers(headers.clone());
        assert_eq!(server.headers, Some(headers));
    }

    #[test]
    fn test_mcp_server_with_header() {
        let server = MCPServerTool::new("test", "https://example.com")
            .with_header("X-First", "one")
            .with_header("X-Second", "two");
        
        let headers = server.headers.unwrap();
        assert_eq!(headers.get("X-First"), Some(&"one".to_string()));
        assert_eq!(headers.get("X-Second"), Some(&"two".to_string()));
    }

    #[test]
    fn test_mcp_server_unique_id() {
        let server = MCPServerTool::new("my-server", "https://example.com");
        assert_eq!(server.unique_id(), "mcp_server:my-server");
    }

    #[test]
    fn test_mcp_server_label() {
        let server = MCPServerTool::new("my-server", "https://example.com");
        assert_eq!(server.label(), "MCP: my-server");
    }

    #[test]
    fn test_mcp_server_builder_chain() {
        let server = MCPServerTool::new("full-test", "https://example.com/mcp")
            .with_auth("secret")
            .with_description("Fully configured server")
            .with_allowed_tools(vec!["read".to_string(), "write".to_string()])
            .with_header("Authorization", "Bearer xyz");
        
        assert_eq!(server.id, "full-test");
        assert_eq!(server.url, "https://example.com/mcp");
        assert_eq!(server.authorization_token, Some("secret".to_string()));
        assert_eq!(server.description, Some("Fully configured server".to_string()));
        assert!(server.allowed_tools.is_some());
        assert!(server.headers.is_some());
    }

    #[test]
    fn test_mcp_server_serialize() {
        let server = MCPServerTool::new("test", "https://example.com")
            .with_description("Test server");
        
        let json = serde_json::to_string(&server).unwrap();
        assert!(json.contains("\"id\":\"test\""));
        assert!(json.contains("\"url\":\"https://example.com\""));
        assert!(json.contains("\"kind\":\"mcp_server\""));
        assert!(json.contains("\"description\":\"Test server\""));
        // Optional None fields should be skipped
        assert!(!json.contains("authorization_token"));
        assert!(!json.contains("allowed_tools"));
        assert!(!json.contains("headers"));
    }

    #[test]
    fn test_mcp_server_deserialize() {
        let json = r#"{
            "id": "test",
            "url": "https://example.com",
            "kind": "mcp_server",
            "description": "Test server"
        }"#;
        
        let server: MCPServerTool = serde_json::from_str(json).unwrap();
        assert_eq!(server.id, "test");
        assert_eq!(server.url, "https://example.com");
        assert_eq!(server.kind, "mcp_server");
        assert_eq!(server.description, Some("Test server".to_string()));
        assert!(server.authorization_token.is_none());
    }
}
