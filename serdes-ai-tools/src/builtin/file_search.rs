//! File search tool using vector similarity search.
//!
//! This module provides a configurable file search tool that uses
//! embeddings for semantic search over files.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{
    definition::ToolDefinition,
    errors::ToolError,
    return_types::{ToolResult, ToolReturn},
    schema::SchemaBuilder,
    tool::Tool,
    RunContext,
};

/// Configuration for the file search tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchConfig {
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum similarity score (0.0 to 1.0).
    pub min_score: f64,
    /// File extensions to search.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub file_extensions: Vec<String>,
    /// Directories to search.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub search_paths: Vec<String>,
    /// Whether to include file content in results.
    pub include_content: bool,
    /// Maximum content snippet length.
    pub max_content_length: usize,
}

impl Default for FileSearchConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            min_score: 0.5,
            file_extensions: Vec::new(),
            search_paths: Vec::new(),
            include_content: true,
            max_content_length: 500,
        }
    }
}

impl FileSearchConfig {
    /// Create a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max results.
    #[must_use]
    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Set minimum similarity score.
    #[must_use]
    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set file extensions to search.
    #[must_use]
    pub fn file_extensions(mut self, exts: Vec<String>) -> Self {
        self.file_extensions = exts;
        self
    }

    /// Add a file extension.
    #[must_use]
    pub fn add_extension(mut self, ext: impl Into<String>) -> Self {
        self.file_extensions.push(ext.into());
        self
    }

    /// Set search paths.
    #[must_use]
    pub fn search_paths(mut self, paths: Vec<String>) -> Self {
        self.search_paths = paths;
        self
    }

    /// Add a search path.
    #[must_use]
    pub fn add_path(mut self, path: impl Into<String>) -> Self {
        self.search_paths.push(path.into());
        self
    }

    /// Set whether to include content.
    #[must_use]
    pub fn include_content(mut self, include: bool) -> Self {
        self.include_content = include;
        self
    }

    /// Set max content length.
    #[must_use]
    pub fn max_content_length(mut self, length: usize) -> Self {
        self.max_content_length = length;
        self
    }
}

/// A file search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// File path.
    pub path: String,
    /// File name.
    pub filename: String,
    /// Similarity score.
    pub score: f64,
    /// Content snippet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Line number where match was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_number: Option<usize>,
}

/// File search tool.
///
/// This tool allows agents to search files using semantic similarity.
/// It requires integration with an embedding model and vector store.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_tools::builtin::{FileSearchTool, FileSearchConfig};
///
/// let tool = FileSearchTool::with_config(
///     FileSearchConfig::new()
///         .max_results(5)
///         .add_extension("rs")
///         .add_extension("py")
/// );
/// ```
pub struct FileSearchTool {
    config: FileSearchConfig,
}

impl FileSearchTool {
    /// Create a new file search tool with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: FileSearchConfig::default(),
        }
    }

    /// Create with a specific config.
    #[must_use]
    pub fn with_config(config: FileSearchConfig) -> Self {
        Self { config }
    }

    /// Get the tool schema.
    fn schema() -> JsonValue {
        SchemaBuilder::new()
            .string("query", "The search query", true)
            .string_array(
                "file_extensions",
                "File extensions to filter by (e.g., ['rs', 'py'])",
                false,
            )
            .integer_constrained(
                "max_results",
                "Maximum number of results to return",
                false,
                Some(1),
                Some(50),
            )
            .build()
            .expect("SchemaBuilder JSON serialization failed")
    }

    /// Perform the search (stub - integrate with vector store).
    async fn search(
        &self,
        query: &str,
        _extensions: &[String],
        max_results: usize,
    ) -> Vec<FileSearchResult> {
        // This is a stub implementation.
        // In a real implementation, you would:
        // 1. Generate an embedding for the query
        // 2. Search a vector store (e.g., Qdrant, Pinecone, etc.)
        // 3. Filter by file extensions if provided
        // 4. Return the top results with content snippets

        vec![FileSearchResult {
            path: "/example/path/file.rs".to_string(),
            filename: "file.rs".to_string(),
            score: 0.95,
            content: Some(format!(
                "This is a placeholder. Integrate with a vector store to get real results. \
                 Query: '{}', Max results: {}",
                query, max_results
            )),
            line_number: Some(1),
        }]
    }
}

impl Default for FileSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> Tool<Deps> for FileSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("file_search", "Search files using semantic similarity")
            .with_parameters(Self::schema())
    }

    async fn call(&self, _ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        let query = args.get("query").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::validation_error(
                "file_search",
                Some("query".to_string()),
                "Missing 'query' field",
            )
        })?;

        if query.trim().is_empty() {
            return Err(ToolError::validation_error(
                "file_search",
                Some("query".to_string()),
                "Query cannot be empty",
            ));
        }

        let extensions: Vec<String> = args
            .get("file_extensions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_else(|| self.config.file_extensions.clone());

        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(self.config.max_results)
            .min(50);

        let results = self.search(query, &extensions, max_results).await;

        // Filter by min score
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| r.score >= self.config.min_score)
            .collect();

        let output = serde_json::json!({
            "query": query,
            "results": filtered,
            "total": filtered.len()
        });

        Ok(ToolReturn::json(output))
    }

    fn max_retries(&self) -> Option<u32> {
        Some(2)
    }
}

impl std::fmt::Debug for FileSearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSearchTool")
            .field("config", &self.config)
            .finish()
    }
}

/// Trait for file search providers.
#[allow(async_fn_in_trait)]
pub trait FileSearchProvider: Send + Sync {
    /// Search files.
    async fn search(
        &self,
        query: &str,
        config: &FileSearchConfig,
    ) -> Result<Vec<FileSearchResult>, ToolError>;
}

/// File indexer for building search indices.
#[allow(async_fn_in_trait)]
pub trait FileIndexer: Send + Sync {
    /// Index files at the given paths.
    async fn index_files(&self, paths: &[String]) -> Result<usize, ToolError>;

    /// Re-index a single file.
    async fn reindex_file(&self, path: &str) -> Result<(), ToolError>;

    /// Remove a file from the index.
    async fn remove_file(&self, path: &str) -> Result<(), ToolError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_search_config() {
        let config = FileSearchConfig::new()
            .max_results(5)
            .min_score(0.7)
            .add_extension("rs")
            .add_path("/src");

        assert_eq!(config.max_results, 5);
        assert_eq!(config.min_score, 0.7);
        assert_eq!(config.file_extensions, vec!["rs"]);
        assert_eq!(config.search_paths, vec!["/src"]);
    }

    #[test]
    fn test_min_score_clamping() {
        let config = FileSearchConfig::new().min_score(1.5);
        assert_eq!(config.min_score, 1.0);

        let config2 = FileSearchConfig::new().min_score(-0.5);
        assert_eq!(config2.min_score, 0.0);
    }

    #[test]
    fn test_file_search_tool_definition() {
        let tool = FileSearchTool::new();
        let def = <FileSearchTool as Tool<()>>::definition(&tool);
        assert_eq!(def.name, "file_search");
        let required = def
            .parameters()
            .get("required")
            .and_then(|value| value.as_array())
            .unwrap();
        assert!(required.iter().any(|value| value.as_str() == Some("query")));
    }

    #[tokio::test]
    async fn test_file_search_tool_call() {
        let tool = FileSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(&ctx, serde_json::json!({"query": "find user auth"}))
            .await
            .unwrap();

        assert!(!result.is_error());
        let json = result.as_json().unwrap();
        assert_eq!(json["query"], "find user auth");
    }

    #[tokio::test]
    async fn test_file_search_missing_query() {
        let tool = FileSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool.call(&ctx, serde_json::json!({})).await;
        assert!(matches!(result, Err(ToolError::ValidationFailed { .. })));
    }

    #[tokio::test]
    async fn test_file_search_empty_query() {
        let tool = FileSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool.call(&ctx, serde_json::json!({"query": "  "})).await;
        assert!(matches!(result, Err(ToolError::ValidationFailed { .. })));
    }

    #[tokio::test]
    async fn test_file_search_with_extensions() {
        let tool = FileSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(
                &ctx,
                serde_json::json!({
                    "query": "test",
                    "file_extensions": ["rs", "py"]
                }),
            )
            .await
            .unwrap();

        assert!(!result.is_error());
    }

    #[test]
    fn test_file_search_result() {
        let result = FileSearchResult {
            path: "/test/file.rs".to_string(),
            filename: "file.rs".to_string(),
            score: 0.95,
            content: Some("fn main() {}".to_string()),
            line_number: Some(1),
        };

        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["filename"], "file.rs");
        assert_eq!(json["score"], 0.95);
    }
}
