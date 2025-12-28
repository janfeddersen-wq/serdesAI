//! DuckDuckGo search tool.
//!
//! This tool provides web search functionality using DuckDuckGo's Instant Answer API.
//! No API key is required, making it perfect for quick prototyping.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_tools::common::DuckDuckGoTool;
//! use serdes_ai_tools::{Tool, RunContext};
//!
//! let tool = DuckDuckGoTool::new();
//! let ctx = RunContext::minimal("test");
//! let result = tool.call(&ctx, serde_json::json!({"query": "rust programming"})).await?;
//! ```
//!
//! ## Limitations
//!
//! The Instant Answer API provides quick answers and related topics, but may not
//! return results for all queries. For more comprehensive search results, consider
//! using the Tavily tool.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{
    definition::ToolDefinition,
    return_types::{ToolResult, ToolReturn},
    schema::SchemaBuilder,
    RunContext, ToolError,
};

/// Configuration for the DuckDuckGo search tool.
#[derive(Debug, Clone)]
pub struct DuckDuckGoConfig {
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Whether to include abstracts in results.
    pub include_abstract: bool,
    /// Whether to include related topics.
    pub include_related: bool,
}

impl Default for DuckDuckGoConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            timeout_secs: 10,
            include_abstract: true,
            include_related: true,
        }
    }
}

impl DuckDuckGoConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Enable or disable abstract inclusion.
    #[must_use]
    pub fn with_abstract(mut self, include: bool) -> Self {
        self.include_abstract = include;
        self
    }

    /// Enable or disable related topics.
    #[must_use]
    pub fn with_related(mut self, include: bool) -> Self {
        self.include_related = include;
        self
    }
}

/// A single search result from DuckDuckGo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuckDuckGoResult {
    /// The title of the result.
    pub title: String,
    /// URL of the result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Text snippet/description.
    pub text: String,
    /// Source of the information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl DuckDuckGoResult {
    /// Create a new result.
    #[must_use]
    pub fn new(title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            url: None,
            text: text.into(),
            source: None,
        }
    }

    /// Set the URL.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set the source.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// DuckDuckGo Instant Answer API response.
#[derive(Debug, Deserialize)]
struct DdgResponse {
    /// Abstract text.
    #[serde(rename = "Abstract")]
    abstract_text: Option<String>,
    /// Abstract source.
    #[serde(rename = "AbstractSource")]
    abstract_source: Option<String>,
    /// Abstract URL.
    #[serde(rename = "AbstractURL")]
    abstract_url: Option<String>,
    /// Heading/title.
    #[serde(rename = "Heading")]
    heading: Option<String>,
    /// Related topics.
    #[serde(rename = "RelatedTopics", default)]
    related_topics: Vec<DdgRelatedTopic>,
    /// Answer (for instant answers).
    #[serde(rename = "Answer")]
    answer: Option<String>,
    /// Definition.
    #[serde(rename = "Definition")]
    definition: Option<String>,
    /// Definition source.
    #[serde(rename = "DefinitionSource")]
    definition_source: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum DdgRelatedTopic {
    /// A single topic.
    Topic {
        #[serde(rename = "Text")]
        text: Option<String>,
        #[serde(rename = "FirstURL")]
        first_url: Option<String>,
    },
    /// A group of topics (category).
    Group {
        #[serde(rename = "Name")]
        name: Option<String>,
        #[serde(rename = "Topics", default)]
        topics: Vec<DdgRelatedTopic>,
    },
}

/// DuckDuckGo search tool.
///
/// Uses DuckDuckGo's Instant Answer API to provide search results.
/// No API key required.
#[derive(Debug, Clone)]
pub struct DuckDuckGoTool {
    config: DuckDuckGoConfig,
    client: Client,
}

impl DuckDuckGoTool {
    /// Create a new DuckDuckGo tool with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DuckDuckGoConfig::default())
    }

    /// Create a new DuckDuckGo tool with custom configuration.
    #[must_use]
    pub fn with_config(config: DuckDuckGoConfig) -> Self {
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

    /// Execute the search.
    async fn search(&self, query: &str) -> Result<Vec<DuckDuckGoResult>, ToolError> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
            urlencoding::encode(query)
        );

        let response = self
            .client
            .get(&url)
            .header("User-Agent", "serdes-ai-tools/0.1")
            .send()
            .await
            .map_err(|e| ToolError::execution_failed(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ToolError::execution_failed(format!(
                "DuckDuckGo API returned status: {}",
                response.status()
            )));
        }

        let ddg_response: DdgResponse = response
            .json()
            .await
            .map_err(|e| ToolError::execution_failed(format!("Failed to parse response: {e}")))?;

        let mut results = Vec::new();

        // Add instant answer if available
        if let Some(answer) = ddg_response.answer.filter(|a| !a.is_empty()) {
            results.push(DuckDuckGoResult::new("Instant Answer", answer));
        }

        // Add definition if available
        if let Some(def) = ddg_response.definition.filter(|d| !d.is_empty()) {
            let mut result = DuckDuckGoResult::new("Definition", def);
            if let Some(source) = ddg_response.definition_source {
                result = result.with_source(source);
            }
            results.push(result);
        }

        // Add abstract if available and configured
        if self.config.include_abstract {
            if let Some(abstract_text) = ddg_response.abstract_text.filter(|a| !a.is_empty()) {
                let title = ddg_response
                    .heading
                    .unwrap_or_else(|| "Summary".to_string());
                let mut result = DuckDuckGoResult::new(title, abstract_text);
                if let Some(url) = ddg_response.abstract_url {
                    result = result.with_url(url);
                }
                if let Some(source) = ddg_response.abstract_source {
                    result = result.with_source(source);
                }
                results.push(result);
            }
        }

        // Add related topics if configured
        if self.config.include_related {
            self.extract_related_topics(&ddg_response.related_topics, &mut results);
        }

        // Limit results
        results.truncate(self.config.max_results);

        Ok(results)
    }

    /// Extract related topics recursively.
    fn extract_related_topics(
        &self,
        topics: &[DdgRelatedTopic],
        results: &mut Vec<DuckDuckGoResult>,
    ) {
        for topic in topics {
            if results.len() >= self.config.max_results {
                break;
            }

            match topic {
                DdgRelatedTopic::Topic { text, first_url } => {
                    if let Some(text) = text.as_ref().filter(|t| !t.is_empty()) {
                        let mut result = DuckDuckGoResult::new("Related", text.clone());
                        if let Some(url) = first_url {
                            result = result.with_url(url.clone());
                        }
                        results.push(result);
                    }
                }
                DdgRelatedTopic::Group { topics, .. } => {
                    self.extract_related_topics(topics, results);
                }
            }
        }
    }
}

impl Default for DuckDuckGoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> crate::Tool<Deps> for DuckDuckGoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "duckduckgo_search",
            "Search the web using DuckDuckGo. Returns instant answers, definitions, and related topics.",
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
                .build()
                .expect("SchemaBuilder JSON serialization failed"),
        )
    }

    async fn call(&self, _ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        let query = args["query"].as_str().ok_or_else(|| {
            ToolError::validation_error(
                "duckduckgo_search",
                Some("query".to_string()),
                "Missing required 'query' parameter",
            )
        })?;

        if query.trim().is_empty() {
            return Err(ToolError::validation_error(
                "duckduckgo_search",
                Some("query".to_string()),
                "Query cannot be empty",
            ));
        }

        // Override max_results if provided
        let max_results = args["max_results"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(self.config.max_results);

        let mut tool = self.clone();
        tool.config.max_results = max_results;

        let results = tool.search(query).await?;

        if results.is_empty() {
            return Ok(ToolReturn::json(serde_json::json!({
                "query": query,
                "results": [],
                "message": "No results found for this query. Try a different search term."
            })));
        }

        Ok(ToolReturn::json(serde_json::json!({
            "query": query,
            "results": results,
            "count": results.len()
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DuckDuckGoConfig::default();
        assert_eq!(config.max_results, 5);
        assert_eq!(config.timeout_secs, 10);
        assert!(config.include_abstract);
        assert!(config.include_related);
    }

    #[test]
    fn test_config_builder() {
        let config = DuckDuckGoConfig::new()
            .with_max_results(10)
            .with_timeout(30)
            .with_abstract(false);

        assert_eq!(config.max_results, 10);
        assert_eq!(config.timeout_secs, 30);
        assert!(!config.include_abstract);
    }

    #[test]
    fn test_result_builder() {
        let result = DuckDuckGoResult::new("Test", "Description")
            .with_url("https://example.com")
            .with_source("Wikipedia");

        assert_eq!(result.title, "Test");
        assert_eq!(result.url, Some("https://example.com".to_string()));
        assert_eq!(result.source, Some("Wikipedia".to_string()));
    }

    #[test]
    fn test_tool_definition() {
        let tool = DuckDuckGoTool::new();
        let def = crate::Tool::<()>::definition(&tool);

        assert_eq!(def.name, "duckduckgo_search");
        assert!(def.description.contains("DuckDuckGo"));
    }

    #[tokio::test]
    async fn test_empty_query_error() {
        let tool = DuckDuckGoTool::new();
        let ctx = crate::RunContext::<()>::minimal("test");

        let result = crate::Tool::call(&tool, &ctx, serde_json::json!({"query": ""})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_missing_query_error() {
        let tool = DuckDuckGoTool::new();
        let ctx = crate::RunContext::<()>::minimal("test");

        let result = crate::Tool::call(&tool, &ctx, serde_json::json!({})).await;
        assert!(result.is_err());
    }
}
