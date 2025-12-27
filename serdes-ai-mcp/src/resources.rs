//! MCP resource handling.
//!
//! This module provides utilities for working with MCP resources.

use crate::error::{McpError, McpResult};
use crate::types::{ResourceContent, ResourceTemplate};
use base64::Engine;
use std::collections::HashMap;
use std::path::Path;
use url::Url;

/// Resource manager for caching and resolving resources.
#[derive(Debug, Default)]
pub struct ResourceManager {
    templates: HashMap<String, ResourceTemplate>,
    cache: HashMap<String, ResourceContent>,
}

impl ResourceManager {
    /// Create a new resource manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a resource template.
    pub fn register_template(&mut self, template: ResourceTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Get all registered templates.
    pub fn templates(&self) -> Vec<&ResourceTemplate> {
        self.templates.values().collect()
    }

    /// Cache a resource.
    pub fn cache(&mut self, uri: impl Into<String>, content: ResourceContent) {
        self.cache.insert(uri.into(), content);
    }

    /// Get a cached resource.
    pub fn get_cached(&self, uri: &str) -> Option<&ResourceContent> {
        self.cache.get(uri)
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Check if a URI is cached.
    pub fn is_cached(&self, uri: &str) -> bool {
        self.cache.contains_key(uri)
    }
}

/// Parse a resource URI.
pub fn parse_resource_uri(uri: &str) -> McpResult<ResourceUri> {
    let parsed = Url::parse(uri).map_err(|e| McpError::Other(format!("Invalid URI: {}", e)))?;

    Ok(ResourceUri {
        scheme: parsed.scheme().to_string(),
        path: parsed.path().to_string(),
        query: parsed
            .query_pairs()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        fragment: parsed.fragment().map(String::from),
    })
}

/// Parsed resource URI.
#[derive(Debug, Clone)]
pub struct ResourceUri {
    /// URI scheme (file, http, etc.).
    pub scheme: String,
    /// Path component.
    pub path: String,
    /// Query parameters.
    pub query: Vec<(String, String)>,
    /// Fragment.
    pub fragment: Option<String>,
}

impl ResourceUri {
    /// Check if this is a file URI.
    pub fn is_file(&self) -> bool {
        self.scheme == "file"
    }

    /// Get as file path (if file URI).
    pub fn as_path(&self) -> Option<&str> {
        if self.is_file() {
            Some(&self.path)
        } else {
            None
        }
    }
}

/// Read a file resource.
pub async fn read_file_resource(path: impl AsRef<Path>) -> McpResult<ResourceContent> {
    let path = path.as_ref();
    let uri = format!("file://{}", path.display());

    let content = tokio::fs::read(path)
        .await
        .map_err(|e| McpError::Other(format!("Failed to read file: {}", e)))?;

    // Detect MIME type
    let mime_type = detect_mime_type(path);

    if is_text_mime(&mime_type) {
        let text = String::from_utf8(content)
            .map_err(|e| McpError::Other(format!("Invalid UTF-8: {}", e)))?;
        Ok(ResourceContent {
            uri,
            mime_type: Some(mime_type),
            text: Some(text),
            blob: None,
        })
    } else {
        Ok(ResourceContent {
            uri,
            mime_type: Some(mime_type),
            text: None,
            blob: Some(base64::engine::general_purpose::STANDARD.encode(&content)),
        })
    }
}

/// Detect MIME type from file extension.
fn detect_mime_type(path: &Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("txt") => "text/plain".to_string(),
        Some("md") => "text/markdown".to_string(),
        Some("json") => "application/json".to_string(),
        Some("html") | Some("htm") => "text/html".to_string(),
        Some("css") => "text/css".to_string(),
        Some("js") => "text/javascript".to_string(),
        Some("ts") => "text/typescript".to_string(),
        Some("rs") => "text/rust".to_string(),
        Some("py") => "text/python".to_string(),
        Some("png") => "image/png".to_string(),
        Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
        Some("gif") => "image/gif".to_string(),
        Some("svg") => "image/svg+xml".to_string(),
        Some("pdf") => "application/pdf".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

/// Check if a MIME type is text-based.
fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/")
        || mime == "application/json"
        || mime == "application/javascript"
        || mime == "application/typescript"
        || mime == "application/xml"
}

/// Resource subscription.
#[derive(Debug, Clone)]
pub struct ResourceSubscription {
    /// Resource URI.
    pub uri: String,
    /// Callback ID.
    pub callback_id: String,
}

/// Resource change event.
#[derive(Debug, Clone)]
pub enum ResourceChange {
    /// Resource was updated.
    Updated {
        /// Resource URI.
        uri: String,
        /// New content.
        content: ResourceContent,
    },
    /// Resource was deleted.
    Deleted {
        /// Resource URI.
        uri: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_resource_uri() {
        let uri = parse_resource_uri("file:///home/user/test.txt").unwrap();
        assert_eq!(uri.scheme, "file");
        assert_eq!(uri.path, "/home/user/test.txt");
        assert!(uri.is_file());
    }

    #[test]
    fn test_parse_resource_uri_with_query() {
        let uri = parse_resource_uri("http://example.com/api?key=value").unwrap();
        assert_eq!(uri.scheme, "http");
        assert_eq!(uri.query.len(), 1);
        assert_eq!(uri.query[0], ("key".to_string(), "value".to_string()));
    }

    #[test]
    fn test_resource_manager() {
        let mut manager = ResourceManager::new();

        let template = ResourceTemplate {
            uri_template: "file:///{path}".to_string(),
            name: "files".to_string(),
            description: Some("Local files".to_string()),
            mime_type: None,
        };
        manager.register_template(template);

        assert_eq!(manager.templates().len(), 1);
    }

    #[test]
    fn test_resource_cache() {
        let mut manager = ResourceManager::new();

        let content = ResourceContent::text("file:///test.txt", "Hello");
        manager.cache("file:///test.txt", content);

        assert!(manager.is_cached("file:///test.txt"));
        assert!(manager.get_cached("file:///test.txt").is_some());
    }

    #[test]
    fn test_detect_mime_type() {
        assert_eq!(detect_mime_type(Path::new("test.txt")), "text/plain");
        assert_eq!(detect_mime_type(Path::new("test.json")), "application/json");
        assert_eq!(detect_mime_type(Path::new("test.png")), "image/png");
        assert_eq!(detect_mime_type(Path::new("test.rs")), "text/rust");
    }

    #[test]
    fn test_is_text_mime() {
        assert!(is_text_mime("text/plain"));
        assert!(is_text_mime("text/html"));
        assert!(is_text_mime("application/json"));
        assert!(!is_text_mime("image/png"));
        assert!(!is_text_mime("application/pdf"));
    }

    #[tokio::test]
    async fn test_read_file_resource() {
        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("mcp_test_resource.txt");
        tokio::fs::write(&temp_file, "Hello, MCP!").await.unwrap();

        let content = read_file_resource(&temp_file).await.unwrap();
        assert_eq!(content.text, Some("Hello, MCP!".to_string()));
        assert!(content.blob.is_none());

        // Cleanup
        tokio::fs::remove_file(&temp_file).await.ok();
    }
}
