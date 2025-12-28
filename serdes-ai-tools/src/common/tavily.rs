//! Tavily search tool.
//!
//! This tool provides AI-optimized web search functionality using Tavily's API.
//! Tavily is designed specifically for AI agents, providing clean, structured
//! search results optimized for LLM consumption.
//!
//! ## API Key
//!
//! Requires a Tavily API key. Get one at: https://tavily.com
//!
//! The API key can be provided directly or via the `TAVILY_API_KEY` environment variable.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_tools::common::TavilyTool;
//! use serdes_ai_tools::{Tool, RunContext};
//!
//! // Using explicit API key
//! let tool = TavilyTool::new("your-api-key");
//!
//! // Using environment variable
//! let tool = TavilyTool::from_env().expect("TAVILY_API_KEY not set");
//!
//! let ctx = RunContext::minimal("test");
//! let result = tool.call(&ctx, serde_json::json!({"query": "latest AI news"})).await?;
//! ```

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::env;

use crate::{
    definition::ToolDefinition,
    return_types::{ToolResult, ToolReturn},
    schema::SchemaBuilder,
    RunContext, ToolError,
};

/// Search depth for Tavily queries.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TavilySearchDepth {
    /// Basic search (faster, less comprehensive).
    #[default]
    Basic,
    /// Advanced search (slower, more comprehensive).
    Advanced,
}

impl TavilySearchDepth {
    /// Get the API value for this depth.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::Advanced => "advanced",
        }
    }
}

/// Configuration for the Tavily search tool.
#[derive(Debug, Clone)]
pub struct TavilyConfig {
    /// API key for Tavily.
    pub api_key: String,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Search depth (basic or advanced).
    pub search_depth: TavilySearchDepth,
    /// Whether to include raw content in results.
    pub include_raw_content: bool,
    /// Whether to include images in results.
    pub include_images: bool,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Domains to include in search (optional filter).
    pub include_domains: Vec<String>,
    /// Domains to exclude from search.
    pub exclude_domains: Vec<String>,
}

impl TavilyConfig {
    /// Create a new configuration with an API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_results: 5,
            search_depth: TavilySearchDepth::Basic,
            include_raw_content: false,
            include_images: false,
            timeout_secs: 30,
            include_domains: Vec::new(),
            exclude_domains: Vec::new(),
        }
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Set the search depth.
    #[must_use]
    pub fn with_search_depth(mut self, depth: TavilySearchDepth) -> Self {
        self.search_depth = depth;
        self
    }

    /// Enable or disable raw content inclusion.
    #[must_use]
    pub fn with_raw_content(mut self, include: bool) -> Self {
        self.include_raw_content = include;
        self
    }

    /// Enable or disable image inclusion.
    #[must_use]
    pub fn with_images(mut self, include: bool) -> Self {
        self.include_images = include;
        self
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Add domains to include.
    #[must_use]
    pub fn with_include_domains(mut self, domains: Vec<String>) -> Self {
        self.include_domains = domains;
        self
    }

    /// Add domains to exclude.
    #[must_use]
    pub fn with_exclude_domains(mut self, domains: Vec<String>) -> Self {
        self.exclude_domains = domains;
        self
    }
}

/// A single search result from Tavily.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TavilyResult {
    /// The title of the result.
    pub title: String,
    /// URL of the result.
    pub url: String,
    /// Content snippet/description.
    pub content: String,
    /// Relevance score (0-1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Raw content (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content: Option<String>,
    /// Published date (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_date: Option<String>,
}

impl TavilyResult {
    /// Create a new result.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            content: content.into(),
            score: None,
            raw_content: None,
            published_date: None,
        }
    }

    /// Set the relevance score.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }
}

/// Tavily API response.
#[derive(Debug, Deserialize)]
struct TavilyResponse {
    /// The search query.
    #[allow(dead_code)]
    query: String,
    /// Search results.
    results: Vec<TavilyApiResult>,
    /// AI-generated answer (if available).
    answer: Option<String>,
    /// Images (if requested).
    #[serde(default)]
    #[allow(dead_code)]
    images: Vec<String>,
    /// Response time in seconds.
    #[allow(dead_code)]
    response_time: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TavilyApiResult {
    title: String,
    url: String,
    content: String,
    score: Option<f64>,
    raw_content: Option<String>,
    published_date: Option<String>,
}

impl From<TavilyApiResult> for TavilyResult {
    fn from(api: TavilyApiResult) -> Self {
        Self {
            title: api.title,
            url: api.url,
            content: api.content,
            score: api.score,
            raw_content: api.raw_content,
            published_date: api.published_date,
        }
    }
}

/// Tavily API error response.
#[derive(Debug, Deserialize)]
struct TavilyErrorResponse {
    detail: Option<String>,
    message: Option<String>,
}

/// Tavily search tool.
///
/// Uses Tavily's AI-optimized search API to provide clean, structured results.
/// Requires an API key from https://tavily.com
#[derive(Debug, Clone)]
pub struct TavilyTool {
    config: TavilyConfig,
    client: Client,
}

impl TavilyTool {
    /// Create a new Tavily tool with an API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_config(TavilyConfig::new(api_key))
    }

    /// Create a new Tavily tool from the `TAVILY_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment variable is not set.
    pub fn from_env() -> Result<Self, ToolError> {
        let api_key = env::var("TAVILY_API_KEY").map_err(|_| {
            ToolError::execution_failed(
                "TAVILY_API_KEY environment variable not set. \
                 Get an API key at https://tavily.com",
            )
        })?;
        Ok(Self::new(api_key))
    }

    /// Create a new Tavily tool with custom configuration.
    #[must_use]
    pub fn with_config(config: TavilyConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .unwrap_or_default();

        Self { config, client }
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.config.max_results = max;
        self
    }

    /// Set the search depth.
    #[must_use]
    pub fn with_search_depth(mut self, depth: TavilySearchDepth) -> Self {
        self.config.search_depth = depth;
        self
    }

    /// Execute the search.
    async fn search(
        &self,
        query: &str,
        max_results: Option<usize>,
        search_depth: Option<TavilySearchDepth>,
    ) -> Result<(Vec<TavilyResult>, Option<String>), ToolError> {
        let max_results = max_results.unwrap_or(self.config.max_results);
        let search_depth = search_depth.unwrap_or(self.config.search_depth);

        let mut request_body = serde_json::json!({
            "api_key": self.config.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth.as_str(),
            "include_raw_content": self.config.include_raw_content,
            "include_images": self.config.include_images,
        });

        // Add optional domain filters
        if !self.config.include_domains.is_empty() {
            request_body["include_domains"] =
                serde_json::to_value(&self.config.include_domains).unwrap();
        }
        if !self.config.exclude_domains.is_empty() {
            request_body["exclude_domains"] =
                serde_json::to_value(&self.config.exclude_domains).unwrap();
        }

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ToolError::execution_failed(format!("HTTP request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let error_body: TavilyErrorResponse = response.json().await.unwrap_or_else(|_| {
                TavilyErrorResponse {
                    detail: None,
                    message: Some(format!("HTTP {status}")),
                }
            });

            let error_msg = error_body
                .detail
                .or(error_body.message)
                .unwrap_or_else(|| format!("Tavily API returned status: {status}"));

            return Err(ToolError::execution_failed(error_msg));
        }

        let tavily_response: TavilyResponse = response
            .json()
            .await
            .map_err(|e| ToolError::execution_failed(format!("Failed to parse response: {e}")))?;

        let results: Vec<TavilyResult> = tavily_response
            .results
            .into_iter()
            .map(TavilyResult::from)
            .collect();

        Ok((results, tavily_response.answer))
    }
}

#[async_trait]
impl<Deps: Send + Sync> crate::Tool<Deps> for TavilyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "tavily_search",
            "Search the web using Tavily's AI-optimized search API. \
             Returns clean, structured results optimized for AI consumption.",
        )
        .with_parameters(
            SchemaBuilder::new()
                .string("query", "The search query", true)
                .integer_constrained(
                    "max_results",
                    "Maximum number of results to return (default: 5)",
                    false,
                    Some(1),
                    Some(20),
                )
                .enum_values(
                    "search_depth",
                    "Search depth: 'basic' (faster) or 'advanced' (more comprehensive)",
                    &["basic", "advanced"],
                    false,
                )
                .build()
                .expect("SchemaBuilder JSON serialization failed"),
        )
    }

    async fn call(&self, _ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        let query = args["query"].as_str().ok_or_else(|| {
            ToolError::validation_error(
                "tavily_search",
                Some("query".to_string()),
                "Missing required 'query' parameter",
            )
        })?;

        if query.trim().is_empty() {
            return Err(ToolError::validation_error(
                "tavily_search",
                Some("query".to_string()),
                "Query cannot be empty",
            ));
        }

        let max_results = args["max_results"].as_u64().map(|n| n as usize);

        let search_depth = args["search_depth"].as_str().and_then(|s| match s {
            "basic" => Some(TavilySearchDepth::Basic),
            "advanced" => Some(TavilySearchDepth::Advanced),
            _ => None,
        });

        let (results, answer) = self.search(query, max_results, search_depth).await?;

        if results.is_empty() && answer.is_none() {
            return Ok(ToolReturn::json(serde_json::json!({
                "query": query,
                "results": [],
                "message": "No results found for this query."
            })));
        }

        let mut response = serde_json::json!({
            "query": query,
            "results": results,
            "count": results.len()
        });

        if let Some(ai_answer) = answer {
            response["answer"] = serde_json::Value::String(ai_answer);
        }

        Ok(ToolReturn::json(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = TavilyConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.max_results, 5);
        assert_eq!(config.search_depth, TavilySearchDepth::Basic);
    }

    #[test]
    fn test_config_builder() {
        let config = TavilyConfig::new("key")
            .with_max_results(10)
            .with_search_depth(TavilySearchDepth::Advanced)
            .with_raw_content(true)
            .with_include_domains(vec!["example.com".into()]);

        assert_eq!(config.max_results, 10);
        assert_eq!(config.search_depth, TavilySearchDepth::Advanced);
        assert!(config.include_raw_content);
        assert_eq!(config.include_domains, vec!["example.com"]);
    }

    #[test]
    fn test_search_depth_as_str() {
        assert_eq!(TavilySearchDepth::Basic.as_str(), "basic");
        assert_eq!(TavilySearchDepth::Advanced.as_str(), "advanced");
    }

    #[test]
    fn test_result_builder() {
        let result = TavilyResult::new("Title", "https://example.com", "Content")
            .with_score(0.95);

        assert_eq!(result.title, "Title");
        assert_eq!(result.url, "https://example.com");
        assert_eq!(result.score, Some(0.95));
    }

    #[test]
    fn test_tool_definition() {
        let tool = TavilyTool::new("test-key");
        let def = crate::Tool::<()>::definition(&tool);

        assert_eq!(def.name, "tavily_search");
        assert!(def.description.contains("Tavily"));
    }

    #[tokio::test]
    async fn test_empty_query_error() {
        let tool = TavilyTool::new("test-key");
        let ctx = crate::RunContext::<()>::minimal("test");

        let result = crate::Tool::call(&tool, &ctx, serde_json::json!({"query": ""})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_missing_query_error() {
        let tool = TavilyTool::new("test-key");
        let ctx = crate::RunContext::<()>::minimal("test");

        let result = crate::Tool::call(&tool, &ctx, serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_from_env_missing() {
        // Ensure env var is not set for this test
        env::remove_var("TAVILY_API_KEY");
        let result = TavilyTool::from_env();
        assert!(result.is_err());
    }
}
