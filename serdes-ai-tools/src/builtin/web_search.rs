//! Web search tool for searching the internet.
//!
//! This module provides a configurable web search tool that can be
//! integrated with various search providers.
//!
//! # Supported Providers
//!
//! - **OpenAI**: Uses `search_context_size` for controlling result depth
//! - **Anthropic/Groq**: Uses `blocked_domains` and `allowed_domains` for filtering
//!
//! # Example
//!
//! ```rust
//! use serdes_ai_tools::builtin::{WebSearchTool, WebSearchConfig, SearchContextSize};
//!
//! let tool = WebSearchTool::builder()
//!     .max_results(5)
//!     .search_context_size(SearchContextSize::High)
//!     .allowed_domains(vec!["docs.rs".to_string(), "crates.io".to_string()])
//!     .build();
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashSet;

use crate::{
    definition::{ObjectJsonSchema, ToolDefinition},
    errors::ToolError,
    return_types::{ToolResult, ToolReturn},
    schema::SchemaBuilder,
    tool::Tool,
    RunContext,
};

/// Errors that can occur during web search configuration or usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSearchError {
    /// Both allowed and blocked domains were specified.
    ConflictingDomainFilters,
    /// An invalid domain was provided.
    InvalidDomain(String),
    /// The domain is not in the allowed list.
    DomainNotAllowed(String),
    /// The domain is in the blocked list.
    DomainBlocked(String),
    /// Maximum uses exceeded.
    MaxUsesExceeded {
        /// Current number of uses.
        current: usize,
        /// Maximum allowed uses.
        max: usize,
    },
}

impl std::fmt::Display for WebSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConflictingDomainFilters => {
                write!(f, "Cannot specify both allowed_domains and blocked_domains")
            }
            Self::InvalidDomain(domain) => {
                write!(f, "Invalid domain: {}", domain)
            }
            Self::DomainNotAllowed(domain) => {
                write!(f, "Domain '{}' is not in the allowed list", domain)
            }
            Self::DomainBlocked(domain) => {
                write!(f, "Domain '{}' is blocked", domain)
            }
            Self::MaxUsesExceeded { current, max } => {
                write!(f, "Maximum uses exceeded: {} of {} allowed", current, max)
            }
        }
    }
}

impl std::error::Error for WebSearchError {}

/// Search context size for OpenAI's web search.
///
/// Controls the amount of context retrieved from web search results.
/// Higher values provide more comprehensive results but may be slower.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextSize {
    /// Minimal context - fastest, least comprehensive.
    Low,
    /// Balanced context - default, good balance of speed and detail.
    #[default]
    Medium,
    /// Maximum context - slowest, most comprehensive.
    High,
}

impl std::fmt::Display for SearchContextSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

/// Configuration for the web search tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Search depth.
    pub search_depth: SearchDepth,
    /// User location for localized results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
    /// Whether to include snippets.
    pub include_snippets: bool,
    /// Whether to include images.
    pub include_images: bool,
    /// API key for the search provider.
    #[serde(skip)]
    pub api_key: Option<String>,
    /// Search context size (OpenAI specific).
    /// Controls the amount of context retrieved from search results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    /// List of allowed domains. If set, only these domains can be searched.
    /// Cannot be used together with `blocked_domains`.
    /// (Anthropic/Groq specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    /// List of blocked domains. These domains will never be searched.
    /// Cannot be used together with `allowed_domains`.
    /// (Anthropic/Groq specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
    /// Maximum number of searches allowed (Anthropic specific).
    /// If None, unlimited searches are allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            search_depth: SearchDepth::default(),
            user_location: None,
            include_snippets: true,
            include_images: false,
            api_key: None,
            search_context_size: None,
            allowed_domains: None,
            blocked_domains: None,
            max_uses: None,
        }
    }
}

impl WebSearchConfig {
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

    /// Set search depth.
    #[must_use]
    pub fn search_depth(mut self, depth: SearchDepth) -> Self {
        self.search_depth = depth;
        self
    }

    /// Set user location.
    #[must_use]
    pub fn user_location(mut self, location: UserLocation) -> Self {
        self.user_location = Some(location);
        self
    }

    /// Set API key.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Enable/disable snippets.
    #[must_use]
    pub fn include_snippets(mut self, include: bool) -> Self {
        self.include_snippets = include;
        self
    }

    /// Enable/disable images.
    #[must_use]
    pub fn include_images(mut self, include: bool) -> Self {
        self.include_images = include;
        self
    }

    /// Set search context size (OpenAI specific).
    #[must_use]
    pub fn search_context_size(mut self, size: SearchContextSize) -> Self {
        self.search_context_size = Some(size);
        self
    }

    /// Set allowed domains.
    ///
    /// # Panics
    ///
    /// Panics if `blocked_domains` is already set.
    #[must_use]
    pub fn allowed_domains(mut self, domains: Vec<String>) -> Self {
        assert!(
            self.blocked_domains.is_none(),
            "Cannot set allowed_domains when blocked_domains is already set"
        );
        self.allowed_domains = Some(domains);
        self
    }

    /// Set blocked domains.
    ///
    /// # Panics
    ///
    /// Panics if `allowed_domains` is already set.
    #[must_use]
    pub fn blocked_domains(mut self, domains: Vec<String>) -> Self {
        assert!(
            self.allowed_domains.is_none(),
            "Cannot set blocked_domains when allowed_domains is already set"
        );
        self.blocked_domains = Some(domains);
        self
    }

    /// Add a single allowed domain.
    ///
    /// # Panics
    ///
    /// Panics if `blocked_domains` is already set.
    #[must_use]
    pub fn allow_domain(mut self, domain: impl Into<String>) -> Self {
        assert!(
            self.blocked_domains.is_none(),
            "Cannot add allowed domain when blocked_domains is already set"
        );
        self.allowed_domains
            .get_or_insert_with(Vec::new)
            .push(domain.into());
        self
    }

    /// Add a single blocked domain.
    ///
    /// # Panics
    ///
    /// Panics if `allowed_domains` is already set.
    #[must_use]
    pub fn block_domain(mut self, domain: impl Into<String>) -> Self {
        assert!(
            self.allowed_domains.is_none(),
            "Cannot add blocked domain when allowed_domains is already set"
        );
        self.blocked_domains
            .get_or_insert_with(Vec::new)
            .push(domain.into());
        self
    }

    /// Set maximum number of uses (Anthropic specific).
    #[must_use]
    pub fn max_uses(mut self, max: u32) -> Self {
        self.max_uses = Some(max);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), WebSearchError> {
        // Check for conflicting domain filters
        if self.allowed_domains.is_some() && self.blocked_domains.is_some() {
            return Err(WebSearchError::ConflictingDomainFilters);
        }

        // Validate domain formats if present
        if let Some(ref domains) = self.allowed_domains {
            for domain in domains {
                validate_domain(domain)?;
            }
        }

        if let Some(ref domains) = self.blocked_domains {
            for domain in domains {
                validate_domain(domain)?;
            }
        }

        Ok(())
    }
}

/// Validate a domain string.
fn validate_domain(domain: &str) -> Result<(), WebSearchError> {
    let domain = domain.trim();

    if domain.is_empty() {
        return Err(WebSearchError::InvalidDomain("empty domain".to_string()));
    }

    // Basic domain validation: should contain only valid characters
    // Allows: alphanumeric, hyphens, dots, and wildcards (*)
    let valid = domain
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '*');

    if !valid {
        return Err(WebSearchError::InvalidDomain(domain.to_string()));
    }

    // Domain should not start or end with dot
    if domain.starts_with('.') || domain.ends_with('.') {
        return Err(WebSearchError::InvalidDomain(domain.to_string()));
    }

    // Each segment should not start or end with hyphen
    for segment in domain.split('.') {
        // Skip wildcard segments
        if segment == "*" {
            continue;
        }
        if segment.is_empty() {
            return Err(WebSearchError::InvalidDomain(domain.to_string()));
        }
        if segment.starts_with('-') || segment.ends_with('-') {
            return Err(WebSearchError::InvalidDomain(domain.to_string()));
        }
    }

    Ok(())
}

/// Normalize a domain for comparison.
fn normalize_domain(domain: &str) -> String {
    domain
        .trim()
        .to_lowercase()
        .strip_prefix("www.")
        .unwrap_or(domain.trim())
        .to_lowercase()
}

/// Check if a domain matches any in the set (supports wildcards).
fn domain_matches(domain: &str, domain_set: &HashSet<String>) -> bool {
    let normalized = normalize_domain(domain);

    // Direct match
    if domain_set.contains(&normalized) {
        return true;
    }

    // Wildcard match (e.g., "*.example.com" matches "sub.example.com")
    for pattern in domain_set {
        if let Some(suffix) = pattern.strip_prefix("*.") {
            if normalized.ends_with(suffix) || normalized == suffix {
                return true;
            }
        }
        // Also check if the domain is a subdomain of an allowed domain
        if normalized.ends_with(&format!(".{}", pattern)) {
            return true;
        }
    }

    false
}

/// Search depth level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchDepth {
    /// Basic search - faster, less comprehensive.
    #[default]
    Basic,
    /// Advanced search - slower, more comprehensive.
    Advanced,
}

impl std::fmt::Display for SearchDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Basic => write!(f, "basic"),
            Self::Advanced => write!(f, "advanced"),
        }
    }
}

/// User location for localized search results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserLocation {
    /// Country code (ISO 3166-1 alpha-2).
    pub country: String,
    /// City name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// Region/state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Timezone (IANA timezone identifier, e.g., "America/New_York").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl UserLocation {
    /// Create a location with just a country.
    #[must_use]
    pub fn country(code: impl Into<String>) -> Self {
        Self {
            country: code.into(),
            city: None,
            region: None,
            timezone: None,
        }
    }

    /// Set the city.
    #[must_use]
    pub fn city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    /// Set the region.
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the timezone (IANA identifier, e.g., "America/New_York").
    #[must_use]
    pub fn timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

/// A search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title.
    pub title: String,
    /// Result URL.
    pub url: String,
    /// Snippet/description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    /// Relevance score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// Web search tool.
///
/// This tool allows agents to search the web for information.
/// It requires integration with an external search provider.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_tools::builtin::{WebSearchTool, WebSearchConfig};
///
/// let tool = WebSearchTool::new()
///     .with_config(WebSearchConfig::new().max_results(5));
/// ```
pub struct WebSearchTool {
    config: WebSearchConfig,
    /// Current use count (runtime state).
    use_count: usize,
}

impl WebSearchTool {
    /// The tool kind identifier.
    pub const KIND: &'static str = "web_search";

    /// Create a new web search tool with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WebSearchConfig::default(),
            use_count: 0,
        }
    }

    /// Create a builder for the web search tool.
    #[must_use]
    pub fn builder() -> WebSearchToolBuilder {
        WebSearchToolBuilder::new()
    }

    /// Create with a specific config.
    #[must_use]
    pub fn with_config(config: WebSearchConfig) -> Self {
        Self {
            config,
            use_count: 0,
        }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &WebSearchConfig {
        &self.config
    }

    /// Get the current use count.
    #[must_use]
    pub fn use_count(&self) -> usize {
        self.use_count
    }

    /// Check if more uses are allowed.
    #[must_use]
    pub fn can_use(&self) -> bool {
        match self.config.max_uses {
            Some(max) => self.use_count < max as usize,
            None => true,
        }
    }

    /// Remaining uses before hitting the limit.
    /// Returns None if unlimited.
    #[must_use]
    pub fn remaining_uses(&self) -> Option<usize> {
        self.config
            .max_uses
            .map(|max| (max as usize).saturating_sub(self.use_count))
    }

    /// Validate a search attempt.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Maximum uses exceeded
    pub fn validate_search(&mut self) -> Result<(), WebSearchError> {
        // Check max uses
        if let Some(max) = self.config.max_uses {
            if self.use_count >= max as usize {
                return Err(WebSearchError::MaxUsesExceeded {
                    current: self.use_count,
                    max: max as usize,
                });
            }
        }

        // Record use
        self.use_count += 1;

        Ok(())
    }

    /// Reset the use counter.
    pub fn reset_use_count(&mut self) {
        self.use_count = 0;
    }

    /// Check if a result URL is allowed based on domain filters.
    #[must_use]
    pub fn is_domain_allowed(&self, url: &str) -> bool {
        let domain = extract_domain(url);

        // Check allowed domains
        if let Some(ref allowed) = self.config.allowed_domains {
            let allowed_set: HashSet<_> = allowed.iter().map(|d| normalize_domain(d)).collect();
            return domain_matches(&domain, &allowed_set);
        }

        // Check blocked domains
        if let Some(ref blocked) = self.config.blocked_domains {
            let blocked_set: HashSet<_> = blocked.iter().map(|d| normalize_domain(d)).collect();
            return !domain_matches(&domain, &blocked_set);
        }

        true
    }

    /// Get the tool schema.
    fn schema() -> ObjectJsonSchema {
        SchemaBuilder::new()
            .string("query", "The search query", true)
            .enum_values(
                "search_depth",
                "Search depth - basic is faster, advanced is more thorough",
                &["basic", "advanced"],
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
    }

    /// Perform the search (stub - integrate with actual provider).
    async fn search(
        &self,
        query: &str,
        _depth: SearchDepth,
        max_results: usize,
    ) -> Vec<SearchResult> {
        // This is a stub implementation.
        // In a real implementation, you would:
        // 1. Call an external search API (e.g., Tavily, Bing, Google, etc.)
        // 2. Parse the results
        // 3. Return structured search results

        vec![SearchResult {
            title: format!("Search results for: {}", query),
            url: "https://example.com/search".to_string(),
            snippet: Some(format!(
                "This is a placeholder. Integrate with a search provider to get real results. \
                 Query: '{}', Max results: {}",
                query, max_results
            )),
            score: Some(1.0),
        }]
    }

    /// Convert to OpenAI provider format.
    #[must_use]
    pub fn to_openai_format(&self) -> JsonValue {
        let mut tool = serde_json::json!({
            "type": "web_search_preview",
        });

        if let Some(ref size) = self.config.search_context_size {
            tool["search_context_size"] = JsonValue::String(size.to_string());
        }

        if let Some(ref location) = self.config.user_location {
            let mut user_location = serde_json::json!({
                "country": location.country,
                "type": "approximate"
            });
            if let Some(ref city) = location.city {
                user_location["city"] = JsonValue::String(city.clone());
            }
            if let Some(ref region) = location.region {
                user_location["region"] = JsonValue::String(region.clone());
            }
            if let Some(ref timezone) = location.timezone {
                user_location["timezone"] = JsonValue::String(timezone.clone());
            }
            tool["user_location"] = user_location;
        }

        tool
    }

    /// Convert to Anthropic provider format.
    #[must_use]
    pub fn to_anthropic_format(&self) -> JsonValue {
        let mut tool = serde_json::json!({
            "type": "web_search",
            "name": "web_search",
        });

        if let Some(max_uses) = self.config.max_uses {
            tool["max_uses"] = JsonValue::from(max_uses);
        }

        if let Some(ref allowed) = self.config.allowed_domains {
            tool["allowed_domains"] = JsonValue::from(allowed.clone());
        }

        if let Some(ref blocked) = self.config.blocked_domains {
            tool["blocked_domains"] = JsonValue::from(blocked.clone());
        }

        if let Some(ref location) = self.config.user_location {
            let mut user_location = serde_json::json!({
                "country": location.country,
            });
            if let Some(ref city) = location.city {
                user_location["city"] = JsonValue::String(city.clone());
            }
            if let Some(ref region) = location.region {
                user_location["region"] = JsonValue::String(region.clone());
            }
            if let Some(ref timezone) = location.timezone {
                user_location["timezone"] = JsonValue::String(timezone.clone());
            }
            tool["user_location"] = user_location;
        }

        tool
    }

    /// Check if this tool is supported by a provider.
    #[must_use]
    pub fn is_supported_by(provider: &str) -> bool {
        matches!(
            provider.to_lowercase().as_str(),
            "openai" | "anthropic" | "claude" | "groq"
        )
    }
}

/// Extract domain from a URL.
fn extract_domain(url: &str) -> String {
    let url = url.trim();

    // Remove protocol
    let without_protocol = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    // Remove path, query, fragment
    let domain = without_protocol
        .split('/')
        .next()
        .unwrap_or(without_protocol);

    // Remove port
    let domain = domain.split(':').next().unwrap_or(domain);

    // Remove www. prefix for normalization
    domain
        .strip_prefix("www.")
        .unwrap_or(domain)
        .to_lowercase()
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for `WebSearchTool`.
#[derive(Debug, Clone, Default)]
pub struct WebSearchToolBuilder {
    config: WebSearchConfig,
}

impl WebSearchToolBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max results.
    #[must_use]
    pub fn max_results(mut self, max: usize) -> Self {
        self.config.max_results = max;
        self
    }

    /// Set search depth.
    #[must_use]
    pub fn search_depth(mut self, depth: SearchDepth) -> Self {
        self.config.search_depth = depth;
        self
    }

    /// Set user location.
    #[must_use]
    pub fn user_location(mut self, location: UserLocation) -> Self {
        self.config.user_location = Some(location);
        self
    }

    /// Set API key.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Enable/disable snippets.
    #[must_use]
    pub fn include_snippets(mut self, include: bool) -> Self {
        self.config.include_snippets = include;
        self
    }

    /// Enable/disable images.
    #[must_use]
    pub fn include_images(mut self, include: bool) -> Self {
        self.config.include_images = include;
        self
    }

    /// Set search context size (OpenAI specific).
    #[must_use]
    pub fn search_context_size(mut self, size: SearchContextSize) -> Self {
        self.config.search_context_size = Some(size);
        self
    }

    /// Set allowed domains.
    #[must_use]
    pub fn allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.config.allowed_domains = Some(domains);
        self.config.blocked_domains = None; // Clear conflicting
        self
    }

    /// Set blocked domains.
    #[must_use]
    pub fn blocked_domains(mut self, domains: Vec<String>) -> Self {
        self.config.blocked_domains = Some(domains);
        self.config.allowed_domains = None; // Clear conflicting
        self
    }

    /// Add a single allowed domain.
    #[must_use]
    pub fn allow_domain(mut self, domain: impl Into<String>) -> Self {
        // Clear blocked domains if set (allowed takes precedence in builder)
        if self.config.blocked_domains.is_some() {
            self.config.blocked_domains = None;
        }
        self.config
            .allowed_domains
            .get_or_insert_with(Vec::new)
            .push(domain.into());
        self
    }

    /// Add a single blocked domain.
    #[must_use]
    pub fn block_domain(mut self, domain: impl Into<String>) -> Self {
        // Clear allowed domains if set (blocked takes precedence in builder)
        if self.config.allowed_domains.is_some() {
            self.config.allowed_domains = None;
        }
        self.config
            .blocked_domains
            .get_or_insert_with(Vec::new)
            .push(domain.into());
        self
    }

    /// Set maximum number of uses (Anthropic specific).
    #[must_use]
    pub fn max_uses(mut self, max: u32) -> Self {
        self.config.max_uses = Some(max);
        self
    }

    /// Build the WebSearchTool.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid.
    #[must_use]
    pub fn build(self) -> WebSearchTool {
        if let Err(e) = self.config.validate() {
            panic!("Invalid WebSearchTool configuration: {}", e);
        }
        WebSearchTool::with_config(self.config)
    }

    /// Try to build the WebSearchTool, returning an error if invalid.
    pub fn try_build(self) -> Result<WebSearchTool, WebSearchError> {
        self.config.validate()?;
        Ok(WebSearchTool::with_config(self.config))
    }
}

#[async_trait]
impl<Deps: Send + Sync> Tool<Deps> for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("web_search", "Search the web for information")
            .with_parameters(Self::schema())
    }

    async fn call(&self, _ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_args("Missing 'query' field"))?;

        if query.trim().is_empty() {
            return Err(ToolError::invalid_args("Query cannot be empty"));
        }

        let search_depth = args
            .get("search_depth")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "advanced" => SearchDepth::Advanced,
                _ => SearchDepth::Basic,
            })
            .unwrap_or(self.config.search_depth);

        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(self.config.max_results)
            .min(50);

        let results = self.search(query, search_depth, max_results).await;

        let output = serde_json::json!({
            "query": query,
            "results": results,
            "total": results.len()
        });

        Ok(ToolReturn::json(output))
    }

    fn max_retries(&self) -> Option<u32> {
        Some(2)
    }
}

impl std::fmt::Debug for WebSearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSearchTool")
            .field("config", &self.config)
            .field("use_count", &self.use_count)
            .finish()
    }
}

/// Trait for web search providers.
#[allow(async_fn_in_trait)]
pub trait WebSearchProvider: Send + Sync {
    /// Perform a search.
    async fn search(
        &self,
        query: &str,
        config: &WebSearchConfig,
    ) -> Result<Vec<SearchResult>, ToolError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_context_size_default() {
        let size = SearchContextSize::default();
        assert_eq!(size, SearchContextSize::Medium);
    }

    #[test]
    fn test_search_context_size_display() {
        assert_eq!(SearchContextSize::Low.to_string(), "low");
        assert_eq!(SearchContextSize::Medium.to_string(), "medium");
        assert_eq!(SearchContextSize::High.to_string(), "high");
    }

    #[test]
    fn test_search_context_size_serde() {
        let json = serde_json::to_string(&SearchContextSize::High).unwrap();
        assert_eq!(json, "\"high\"");

        let parsed: SearchContextSize = serde_json::from_str("\"low\"").unwrap();
        assert_eq!(parsed, SearchContextSize::Low);
    }

    #[test]
    fn test_web_search_config() {
        let config = WebSearchConfig::new()
            .max_results(5)
            .search_depth(SearchDepth::Advanced)
            .include_snippets(false)
            .search_context_size(SearchContextSize::High)
            .max_uses(10);

        assert_eq!(config.max_results, 5);
        assert_eq!(config.search_depth, SearchDepth::Advanced);
        assert!(!config.include_snippets);
        assert_eq!(config.search_context_size, Some(SearchContextSize::High));
        assert_eq!(config.max_uses, Some(10));
    }

    #[test]
    fn test_web_search_config_allowed_domains() {
        let config = WebSearchConfig::new()
            .allow_domain("example.com")
            .allow_domain("docs.rs");

        assert_eq!(
            config.allowed_domains,
            Some(vec!["example.com".to_string(), "docs.rs".to_string()])
        );
        assert!(config.blocked_domains.is_none());
    }

    #[test]
    fn test_web_search_config_blocked_domains() {
        let config = WebSearchConfig::new()
            .block_domain("evil.com")
            .block_domain("spam.net");

        assert_eq!(
            config.blocked_domains,
            Some(vec!["evil.com".to_string(), "spam.net".to_string()])
        );
        assert!(config.allowed_domains.is_none());
    }

    #[test]
    #[should_panic(expected = "Cannot set blocked_domains when allowed_domains is already set")]
    fn test_web_search_config_conflicting_domains_panic() {
        let _ = WebSearchConfig::new()
            .allowed_domains(vec!["example.com".to_string()])
            .blocked_domains(vec!["evil.com".to_string()]);
    }

    #[test]
    fn test_web_search_config_validation() {
        let valid = WebSearchConfig::new().allow_domain("example.com");
        assert!(valid.validate().is_ok());

        // Invalid domain
        let invalid = WebSearchConfig {
            allowed_domains: Some(vec!["invalid domain with spaces".to_string()]),
            ..Default::default()
        };
        assert!(matches!(
            invalid.validate(),
            Err(WebSearchError::InvalidDomain(_))
        ));
    }

    #[test]
    fn test_user_location() {
        let loc = UserLocation::country("US")
            .city("New York")
            .region("NY")
            .timezone("America/New_York");

        assert_eq!(loc.country, "US");
        assert_eq!(loc.city, Some("New York".to_string()));
        assert_eq!(loc.region, Some("NY".to_string()));
        assert_eq!(loc.timezone, Some("America/New_York".to_string()));
    }

    #[test]
    fn test_user_location_serde() {
        let loc = UserLocation::country("GB")
            .city("London")
            .timezone("Europe/London");

        let json = serde_json::to_string(&loc).unwrap();
        assert!(json.contains("\"country\":\"GB\""));
        assert!(json.contains("\"timezone\":\"Europe/London\""));

        let parsed: UserLocation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.country, "GB");
        assert_eq!(parsed.timezone, Some("Europe/London".to_string()));
    }

    #[test]
    fn test_web_search_tool_builder() {
        let tool = WebSearchTool::builder()
            .max_results(5)
            .search_depth(SearchDepth::Advanced)
            .search_context_size(SearchContextSize::High)
            .allow_domain("example.com")
            .max_uses(10)
            .build();

        assert_eq!(tool.config().max_results, 5);
        assert_eq!(tool.config().search_depth, SearchDepth::Advanced);
        assert_eq!(
            tool.config().search_context_size,
            Some(SearchContextSize::High)
        );
        assert_eq!(
            tool.config().allowed_domains,
            Some(vec!["example.com".to_string()])
        );
        assert_eq!(tool.config().max_uses, Some(10));
    }

    #[test]
    fn test_web_search_tool_builder_domain_switching() {
        // Builder should handle switching between allowed/blocked
        let tool = WebSearchTool::builder()
            .allow_domain("good.com")
            .block_domain("evil.com") // This should clear allowed
            .build();

        assert!(tool.config().allowed_domains.is_none());
        assert_eq!(
            tool.config().blocked_domains,
            Some(vec!["evil.com".to_string()])
        );
    }

    #[test]
    fn test_web_search_tool_max_uses() {
        let mut tool = WebSearchTool::builder().max_uses(2).build();

        assert!(tool.can_use());
        assert_eq!(tool.remaining_uses(), Some(2));

        // First use
        assert!(tool.validate_search().is_ok());
        assert_eq!(tool.remaining_uses(), Some(1));

        // Second use
        assert!(tool.validate_search().is_ok());
        assert_eq!(tool.remaining_uses(), Some(0));

        // Third use should fail
        assert!(matches!(
            tool.validate_search(),
            Err(WebSearchError::MaxUsesExceeded { current: 2, max: 2 })
        ));
    }

    #[test]
    fn test_web_search_tool_domain_filtering() {
        let tool = WebSearchTool::builder()
            .allow_domain("example.com")
            .allow_domain("docs.rs")
            .build();

        assert!(tool.is_domain_allowed("https://example.com/page"));
        assert!(tool.is_domain_allowed("https://docs.rs/crate"));
        assert!(tool.is_domain_allowed("https://api.example.com/v1")); // subdomain
        assert!(!tool.is_domain_allowed("https://evil.com"));
    }

    #[test]
    fn test_web_search_tool_blocked_domains() {
        let tool = WebSearchTool::builder()
            .block_domain("evil.com")
            .block_domain("spam.net")
            .build();

        assert!(tool.is_domain_allowed("https://example.com"));
        assert!(!tool.is_domain_allowed("https://evil.com"));
        assert!(!tool.is_domain_allowed("https://sub.evil.com")); // subdomain
    }

    #[test]
    fn test_web_search_tool_reset_use_count() {
        let mut tool = WebSearchTool::builder().max_uses(1).build();

        assert!(tool.validate_search().is_ok());
        assert!(!tool.can_use());

        tool.reset_use_count();
        assert!(tool.can_use());
        assert_eq!(tool.use_count(), 0);
    }

    #[test]
    fn test_web_search_tool_to_openai_format() {
        let tool = WebSearchTool::builder()
            .search_context_size(SearchContextSize::High)
            .user_location(
                UserLocation::country("US")
                    .city("New York")
                    .region("NY")
                    .timezone("America/New_York"),
            )
            .build();

        let format = tool.to_openai_format();

        assert_eq!(format["type"], "web_search_preview");
        assert_eq!(format["search_context_size"], "high");
        assert_eq!(format["user_location"]["country"], "US");
        assert_eq!(format["user_location"]["city"], "New York");
        assert_eq!(format["user_location"]["region"], "NY");
        assert_eq!(format["user_location"]["timezone"], "America/New_York");
    }

    #[test]
    fn test_web_search_tool_to_anthropic_format() {
        let tool = WebSearchTool::builder()
            .max_uses(5)
            .allow_domain("example.com")
            .user_location(UserLocation::country("GB").timezone("Europe/London"))
            .build();

        let format = tool.to_anthropic_format();

        assert_eq!(format["type"], "web_search");
        assert_eq!(format["name"], "web_search");
        assert_eq!(format["max_uses"], 5);
        assert_eq!(format["allowed_domains"][0], "example.com");
        assert_eq!(format["user_location"]["country"], "GB");
        assert_eq!(format["user_location"]["timezone"], "Europe/London");
    }

    #[test]
    fn test_web_search_tool_to_anthropic_format_blocked() {
        let tool = WebSearchTool::builder()
            .block_domain("evil.com")
            .build();

        let format = tool.to_anthropic_format();

        assert_eq!(format["blocked_domains"][0], "evil.com");
        assert!(format.get("allowed_domains").is_none());
    }

    #[test]
    fn test_web_search_tool_is_supported_by() {
        assert!(WebSearchTool::is_supported_by("openai"));
        assert!(WebSearchTool::is_supported_by("OpenAI"));
        assert!(WebSearchTool::is_supported_by("anthropic"));
        assert!(WebSearchTool::is_supported_by("claude"));
        assert!(WebSearchTool::is_supported_by("groq"));

        assert!(!WebSearchTool::is_supported_by("google"));
        assert!(!WebSearchTool::is_supported_by("cohere"));
    }

    #[test]
    fn test_web_search_tool_definition() {
        let tool = WebSearchTool::new();
        let def = <WebSearchTool as Tool<()>>::definition(&tool);
        assert_eq!(def.name, "web_search");
        assert!(def.parameters().is_required("query"));
    }

    #[tokio::test]
    async fn test_web_search_tool_call() {
        let tool = WebSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(&ctx, serde_json::json!({"query": "rust programming"}))
            .await
            .unwrap();

        assert!(!result.is_error());
        let json = result.as_json().unwrap();
        assert_eq!(json["query"], "rust programming");
    }

    #[tokio::test]
    async fn test_web_search_missing_query() {
        let tool = WebSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool.call(&ctx, serde_json::json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn test_web_search_empty_query() {
        let tool = WebSearchTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool.call(&ctx, serde_json::json!({"query": "  "})).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[test]
    fn test_search_depth_display() {
        assert_eq!(SearchDepth::Basic.to_string(), "basic");
        assert_eq!(SearchDepth::Advanced.to_string(), "advanced");
    }

    #[test]
    fn test_web_search_error_display() {
        let err = WebSearchError::ConflictingDomainFilters;
        assert!(err.to_string().contains("Cannot specify both"));

        let err = WebSearchError::InvalidDomain("bad domain".to_string());
        assert!(err.to_string().contains("Invalid domain"));

        let err = WebSearchError::DomainNotAllowed("evil.com".to_string());
        assert!(err.to_string().contains("not in the allowed list"));

        let err = WebSearchError::DomainBlocked("evil.com".to_string());
        assert!(err.to_string().contains("blocked"));

        let err = WebSearchError::MaxUsesExceeded { current: 5, max: 5 };
        assert!(err.to_string().contains("5 of 5"));
    }

    #[test]
    fn test_web_search_builder_try_build() {
        // Valid
        let result = WebSearchTool::builder()
            .allow_domain("example.com")
            .try_build();
        assert!(result.is_ok());

        // Valid with blocked
        let result = WebSearchTool::builder()
            .block_domain("evil.com")
            .try_build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_domain() {
        // Valid domains
        assert!(validate_domain("example.com").is_ok());
        assert!(validate_domain("sub.example.com").is_ok());
        assert!(validate_domain("*.example.com").is_ok());
        assert!(validate_domain("my-domain.com").is_ok());

        // Invalid domains
        assert!(validate_domain("").is_err());
        assert!(validate_domain("has space.com").is_err());
        assert!(validate_domain("-invalid.com").is_err());
        assert!(validate_domain("invalid-.com").is_err());
        assert!(validate_domain(".invalid.com").is_err());
        assert!(validate_domain("invalid.com.").is_err());
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://example.com"), "example.com");
        assert_eq!(extract_domain("http://www.example.com"), "example.com");
        assert_eq!(extract_domain("https://example.com/path"), "example.com");
        assert_eq!(extract_domain("https://example.com:8080"), "example.com");
        assert_eq!(extract_domain("https://sub.example.com"), "sub.example.com");
        assert_eq!(extract_domain("example.com"), "example.com");
    }
}
