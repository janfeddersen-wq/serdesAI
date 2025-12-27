//! Web fetch tool for accessing URL contents.
//!
//! This module provides a configurable web fetch tool that allows agents
//! to access content from URLs. It supports provider-specific implementations
//! for Anthropic and Google.
//!
//! # Supported Providers
//!
//! - **Anthropic**: Uses native URL content fetching
//! - **Google**: Uses native web content retrieval
//!
//! # Example
//!
//! ```rust
//! use serdes_ai_tools::builtin::{WebFetchTool, WebFetchConfig};
//!
//! let tool = WebFetchTool::builder()
//!     .max_uses(10)
//!     .allowed_domains(vec!["example.com".to_string(), "docs.rs".to_string()])
//!     .enable_citations(true)
//!     .build();
//!
//! // Convert to provider-specific format
//! let anthropic_format = tool.to_anthropic_format();
//! let google_format = tool.to_google_format();
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashSet;

/// Errors that can occur during web fetch configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebFetchError {
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

impl std::fmt::Display for WebFetchError {
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

impl std::error::Error for WebFetchError {}

/// Configuration for the web fetch tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WebFetchConfig {
    /// Maximum number of URL fetches allowed.
    /// If None, unlimited fetches are allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<usize>,

    /// List of allowed domains. If set, only these domains can be fetched.
    /// Cannot be used together with `blocked_domains`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,

    /// List of blocked domains. These domains will never be fetched.
    /// Cannot be used together with `allowed_domains`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,

    /// Whether to enable citations for fetched content.
    /// Supported by some providers.
    #[serde(default)]
    pub enable_citations: bool,

    /// Maximum content length in tokens for fetched content.
    /// If None, provider default is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<usize>,
}

impl Default for WebFetchConfig {
    fn default() -> Self {
        Self {
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
            enable_citations: false,
            max_content_tokens: None,
        }
    }
}

impl WebFetchConfig {
    /// Create a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of uses.
    #[must_use]
    pub fn max_uses(mut self, max: usize) -> Self {
        self.max_uses = Some(max);
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

    /// Enable or disable citations.
    #[must_use]
    pub fn enable_citations(mut self, enable: bool) -> Self {
        self.enable_citations = enable;
        self
    }

    /// Set maximum content tokens.
    #[must_use]
    pub fn max_content_tokens(mut self, max: usize) -> Self {
        self.max_content_tokens = Some(max);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), WebFetchError> {
        // Check for conflicting domain filters
        if self.allowed_domains.is_some() && self.blocked_domains.is_some() {
            return Err(WebFetchError::ConflictingDomainFilters);
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
fn validate_domain(domain: &str) -> Result<(), WebFetchError> {
    let domain = domain.trim();
    
    if domain.is_empty() {
        return Err(WebFetchError::InvalidDomain("empty domain".to_string()));
    }

    // Basic domain validation: should contain only valid characters
    // Allows: alphanumeric, hyphens, dots, and wildcards (*)
    let valid = domain.chars().all(|c| {
        c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '*'
    });

    if !valid {
        return Err(WebFetchError::InvalidDomain(domain.to_string()));
    }

    // Domain should not start or end with dot
    if domain.starts_with('.') || domain.ends_with('.') {
        return Err(WebFetchError::InvalidDomain(domain.to_string()));
    }

    // Each segment should not start or end with hyphen
    for segment in domain.split('.') {
        // Skip wildcard segments
        if segment == "*" {
            continue;
        }
        if segment.is_empty() {
            return Err(WebFetchError::InvalidDomain(domain.to_string()));
        }
        if segment.starts_with('-') || segment.ends_with('-') {
            return Err(WebFetchError::InvalidDomain(domain.to_string()));
        }
    }

    Ok(())
}

/// Web fetch tool for accessing URL contents.
///
/// This is a builtin tool that maps to provider-specific implementations.
/// It allows agents to fetch content from URLs with configurable restrictions.
///
/// # Supported Providers
///
/// - **Anthropic**: Maps to Claude's native URL fetching capability
/// - **Google**: Maps to Gemini's web content retrieval
///
/// # Example
///
/// ```rust
/// use serdes_ai_tools::builtin::{WebFetchTool, WebFetchConfig};
///
/// // Create with builder pattern
/// let tool = WebFetchTool::builder()
///     .max_uses(5)
///     .allow_domain("example.com")
///     .allow_domain("docs.rs")
///     .enable_citations(true)
///     .build();
///
/// // Or with config
/// let tool = WebFetchTool::with_config(
///     WebFetchConfig::new()
///         .max_uses(10)
///         .blocked_domains(vec!["evil.com".to_string()])
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFetchTool {
    /// Tool configuration.
    config: WebFetchConfig,
    /// Current use count (not serialized - runtime state).
    #[serde(skip)]
    use_count: usize,
    /// Tool kind identifier.
    kind: String,
}

impl WebFetchTool {
    /// The tool kind identifier.
    pub const KIND: &'static str = "web_fetch";

    /// Create a new web fetch tool with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WebFetchConfig::default(),
            use_count: 0,
            kind: Self::KIND.to_string(),
        }
    }

    /// Create a builder for the web fetch tool.
    #[must_use]
    pub fn builder() -> WebFetchToolBuilder {
        WebFetchToolBuilder::new()
    }

    /// Create with a specific configuration.
    #[must_use]
    pub fn with_config(config: WebFetchConfig) -> Self {
        Self {
            config,
            use_count: 0,
            kind: Self::KIND.to_string(),
        }
    }

    /// Get the tool kind.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &WebFetchConfig {
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
            Some(max) => self.use_count < max,
            None => true,
        }
    }

    /// Remaining uses before hitting the limit.
    /// Returns None if unlimited.
    #[must_use]
    pub fn remaining_uses(&self) -> Option<usize> {
        self.config.max_uses.map(|max| max.saturating_sub(self.use_count))
    }

    /// Validate and record a URL fetch attempt.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Maximum uses exceeded
    /// - Domain is not allowed
    /// - Domain is blocked
    pub fn validate_url(&mut self, url: &str) -> Result<(), WebFetchError> {
        // Check max uses
        if let Some(max) = self.config.max_uses {
            if self.use_count >= max {
                return Err(WebFetchError::MaxUsesExceeded {
                    current: self.use_count,
                    max,
                });
            }
        }

        // Extract domain from URL
        let domain = extract_domain(url);

        // Check domain filters
        if let Some(ref allowed) = self.config.allowed_domains {
            let allowed_set: HashSet<_> = allowed.iter().map(|d| normalize_domain(d)).collect();
            if !domain_matches(&domain, &allowed_set) {
                return Err(WebFetchError::DomainNotAllowed(domain));
            }
        }

        if let Some(ref blocked) = self.config.blocked_domains {
            let blocked_set: HashSet<_> = blocked.iter().map(|d| normalize_domain(d)).collect();
            if domain_matches(&domain, &blocked_set) {
                return Err(WebFetchError::DomainBlocked(domain));
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

    /// Convert to Anthropic provider format.
    ///
    /// Anthropic uses a specific format for builtin tools.
    #[must_use]
    pub fn to_anthropic_format(&self) -> JsonValue {
        let mut tool = serde_json::json!({
            "type": "web_fetch",
        });

        // Anthropic-specific configuration
        if let Some(max_uses) = self.config.max_uses {
            tool["max_uses"] = JsonValue::from(max_uses);
        }

        if let Some(ref allowed) = self.config.allowed_domains {
            tool["allowed_domains"] = JsonValue::from(allowed.clone());
        }

        if let Some(ref blocked) = self.config.blocked_domains {
            tool["blocked_domains"] = JsonValue::from(blocked.clone());
        }

        if self.config.enable_citations {
            tool["enable_citations"] = JsonValue::Bool(true);
        }

        if let Some(max_tokens) = self.config.max_content_tokens {
            tool["max_content_tokens"] = JsonValue::from(max_tokens);
        }

        tool
    }

    /// Convert to Google provider format.
    ///
    /// Google/Gemini uses a different format for web content tools.
    #[must_use]
    pub fn to_google_format(&self) -> JsonValue {
        let mut tool = serde_json::json!({
            "name": "google_web_fetch",
            "type": "retrieval",
        });

        // Build retrieval config
        let mut retrieval_config = serde_json::Map::new();

        if let Some(max_uses) = self.config.max_uses {
            retrieval_config.insert("maxUses".to_string(), JsonValue::from(max_uses));
        }

        // Google uses different naming conventions
        if let Some(ref allowed) = self.config.allowed_domains {
            retrieval_config.insert("allowedDomains".to_string(), JsonValue::from(allowed.clone()));
        }

        if let Some(ref blocked) = self.config.blocked_domains {
            retrieval_config.insert("blockedDomains".to_string(), JsonValue::from(blocked.clone()));
        }

        if self.config.enable_citations {
            retrieval_config.insert("includeCitations".to_string(), JsonValue::Bool(true));
        }

        if let Some(max_tokens) = self.config.max_content_tokens {
            retrieval_config.insert("maxContentTokens".to_string(), JsonValue::from(max_tokens));
        }

        if !retrieval_config.is_empty() {
            tool["retrievalConfig"] = JsonValue::Object(retrieval_config);
        }

        tool
    }

    /// Check if this tool is supported by a provider.
    #[must_use]
    pub fn is_supported_by(provider: &str) -> bool {
        matches!(
            provider.to_lowercase().as_str(),
            "anthropic" | "claude" | "google" | "gemini"
        )
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for WebFetchTool {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config && self.kind == other.kind
    }
}

/// Builder for `WebFetchTool`.
#[derive(Debug, Clone, Default)]
pub struct WebFetchToolBuilder {
    config: WebFetchConfig,
}

impl WebFetchToolBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of uses.
    #[must_use]
    pub fn max_uses(mut self, max: usize) -> Self {
        self.config.max_uses = Some(max);
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

    /// Enable or disable citations.
    #[must_use]
    pub fn enable_citations(mut self, enable: bool) -> Self {
        self.config.enable_citations = enable;
        self
    }

    /// Set maximum content tokens.
    #[must_use]
    pub fn max_content_tokens(mut self, max: usize) -> Self {
        self.config.max_content_tokens = Some(max);
        self
    }

    /// Build the WebFetchTool.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid (though the builder
    /// prevents most invalid states).
    #[must_use]
    pub fn build(self) -> WebFetchTool {
        // Builder design prevents conflicting states, but validate anyway
        if let Err(e) = self.config.validate() {
            panic!("Invalid WebFetchTool configuration: {}", e);
        }
        WebFetchTool::with_config(self.config)
    }

    /// Try to build the WebFetchTool, returning an error if invalid.
    pub fn try_build(self) -> Result<WebFetchTool, WebFetchError> {
        self.config.validate()?;
        Ok(WebFetchTool::with_config(self.config))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_fetch_config_default() {
        let config = WebFetchConfig::default();
        assert!(config.max_uses.is_none());
        assert!(config.allowed_domains.is_none());
        assert!(config.blocked_domains.is_none());
        assert!(!config.enable_citations);
        assert!(config.max_content_tokens.is_none());
    }

    #[test]
    fn test_web_fetch_config_builder() {
        let config = WebFetchConfig::new()
            .max_uses(10)
            .enable_citations(true)
            .max_content_tokens(5000);

        assert_eq!(config.max_uses, Some(10));
        assert!(config.enable_citations);
        assert_eq!(config.max_content_tokens, Some(5000));
    }

    #[test]
    fn test_web_fetch_config_allowed_domains() {
        let config = WebFetchConfig::new()
            .allow_domain("example.com")
            .allow_domain("docs.rs");

        assert_eq!(
            config.allowed_domains,
            Some(vec!["example.com".to_string(), "docs.rs".to_string()])
        );
    }

    #[test]
    fn test_web_fetch_config_blocked_domains() {
        let config = WebFetchConfig::new()
            .block_domain("evil.com")
            .block_domain("malware.net");

        assert_eq!(
            config.blocked_domains,
            Some(vec!["evil.com".to_string(), "malware.net".to_string()])
        );
    }

    #[test]
    #[should_panic(expected = "Cannot set blocked_domains when allowed_domains is already set")]
    fn test_web_fetch_config_conflicting_domains_panic() {
        let _ = WebFetchConfig::new()
            .allowed_domains(vec!["example.com".to_string()])
            .blocked_domains(vec!["evil.com".to_string()]);
    }

    #[test]
    fn test_web_fetch_config_validation() {
        let valid = WebFetchConfig::new().allow_domain("example.com");
        assert!(valid.validate().is_ok());

        // Invalid domain
        let invalid = WebFetchConfig {
            allowed_domains: Some(vec!["invalid domain with spaces".to_string()]),
            ..Default::default()
        };
        assert!(matches!(
            invalid.validate(),
            Err(WebFetchError::InvalidDomain(_))
        ));
    }

    #[test]
    fn test_web_fetch_tool_new() {
        let tool = WebFetchTool::new();
        assert_eq!(tool.kind(), "web_fetch");
        assert_eq!(tool.use_count(), 0);
        assert!(tool.can_use());
    }

    #[test]
    fn test_web_fetch_tool_builder() {
        let tool = WebFetchTool::builder()
            .max_uses(5)
            .allow_domain("example.com")
            .enable_citations(true)
            .max_content_tokens(1000)
            .build();

        assert_eq!(tool.config().max_uses, Some(5));
        assert!(tool.config().enable_citations);
        assert_eq!(tool.config().max_content_tokens, Some(1000));
    }

    #[test]
    fn test_web_fetch_tool_max_uses() {
        let mut tool = WebFetchTool::builder().max_uses(2).build();

        assert!(tool.can_use());
        assert_eq!(tool.remaining_uses(), Some(2));

        // First use
        assert!(tool.validate_url("https://example.com").is_ok());
        assert_eq!(tool.remaining_uses(), Some(1));

        // Second use
        assert!(tool.validate_url("https://example.com/page").is_ok());
        assert_eq!(tool.remaining_uses(), Some(0));

        // Third use should fail
        assert!(matches!(
            tool.validate_url("https://example.com/another"),
            Err(WebFetchError::MaxUsesExceeded { current: 2, max: 2 })
        ));
    }

    #[test]
    fn test_web_fetch_tool_allowed_domains() {
        let mut tool = WebFetchTool::builder()
            .allow_domain("example.com")
            .allow_domain("docs.rs")
            .build();

        // Allowed domain
        assert!(tool.validate_url("https://example.com/page").is_ok());
        assert!(tool.validate_url("https://docs.rs/crate").is_ok());

        // Subdomain of allowed domain
        assert!(tool.validate_url("https://api.example.com/v1").is_ok());

        // Not allowed domain
        assert!(matches!(
            tool.validate_url("https://evil.com"),
            Err(WebFetchError::DomainNotAllowed(_))
        ));
    }

    #[test]
    fn test_web_fetch_tool_blocked_domains() {
        let mut tool = WebFetchTool::builder()
            .block_domain("evil.com")
            .block_domain("malware.net")
            .build();

        // Allowed (not blocked)
        assert!(tool.validate_url("https://example.com").is_ok());

        // Blocked domain
        assert!(matches!(
            tool.validate_url("https://evil.com"),
            Err(WebFetchError::DomainBlocked(_))
        ));

        // Subdomain of blocked domain
        assert!(matches!(
            tool.validate_url("https://sub.evil.com"),
            Err(WebFetchError::DomainBlocked(_))
        ));
    }

    #[test]
    fn test_web_fetch_tool_wildcard_domains() {
        let mut tool = WebFetchTool::builder()
            .allowed_domains(vec!["*.example.com".to_string()])
            .build();

        // Wildcard match
        assert!(tool.validate_url("https://api.example.com").is_ok());
        assert!(tool.validate_url("https://deep.sub.example.com").is_ok());

        // Exact domain also matches
        assert!(tool.validate_url("https://example.com").is_ok());
    }

    #[test]
    fn test_web_fetch_tool_reset_use_count() {
        let mut tool = WebFetchTool::builder().max_uses(1).build();

        assert!(tool.validate_url("https://example.com").is_ok());
        assert!(!tool.can_use());

        tool.reset_use_count();
        assert!(tool.can_use());
        assert_eq!(tool.use_count(), 0);
    }

    #[test]
    fn test_web_fetch_tool_to_anthropic_format() {
        let tool = WebFetchTool::builder()
            .max_uses(10)
            .allow_domain("example.com")
            .enable_citations(true)
            .max_content_tokens(5000)
            .build();

        let format = tool.to_anthropic_format();

        assert_eq!(format["type"], "web_fetch");
        assert_eq!(format["max_uses"], 10);
        assert_eq!(format["allowed_domains"][0], "example.com");
        assert_eq!(format["enable_citations"], true);
        assert_eq!(format["max_content_tokens"], 5000);
    }

    #[test]
    fn test_web_fetch_tool_to_google_format() {
        let tool = WebFetchTool::builder()
            .max_uses(10)
            .block_domain("evil.com")
            .enable_citations(true)
            .build();

        let format = tool.to_google_format();

        assert_eq!(format["name"], "google_web_fetch");
        assert_eq!(format["type"], "retrieval");
        assert_eq!(format["retrievalConfig"]["maxUses"], 10);
        assert_eq!(format["retrievalConfig"]["blockedDomains"][0], "evil.com");
        assert_eq!(format["retrievalConfig"]["includeCitations"], true);
    }

    #[test]
    fn test_web_fetch_tool_minimal_format() {
        let tool = WebFetchTool::new();

        let anthropic = tool.to_anthropic_format();
        assert_eq!(anthropic["type"], "web_fetch");
        assert!(anthropic.get("max_uses").is_none());

        let google = tool.to_google_format();
        assert_eq!(google["type"], "retrieval");
        // No retrievalConfig when empty
        assert!(google.get("retrievalConfig").is_none());
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

    #[test]
    fn test_normalize_domain() {
        assert_eq!(normalize_domain("Example.COM"), "example.com");
        assert_eq!(normalize_domain("www.example.com"), "example.com");
        assert_eq!(normalize_domain("  example.com  "), "example.com");
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
    fn test_domain_matches() {
        let allowed: HashSet<String> = vec![
            "example.com".to_string(),
            "*.docs.rs".to_string(),
        ]
        .into_iter()
        .collect();

        // Direct match
        assert!(domain_matches("example.com", &allowed));
        assert!(domain_matches("EXAMPLE.COM", &allowed));

        // Subdomain match
        assert!(domain_matches("api.example.com", &allowed));

        // Wildcard match
        assert!(domain_matches("docs.rs", &allowed));
        assert!(domain_matches("api.docs.rs", &allowed));

        // No match
        assert!(!domain_matches("other.com", &allowed));
    }

    #[test]
    fn test_web_fetch_tool_serde() {
        let tool = WebFetchTool::builder()
            .max_uses(5)
            .allow_domain("example.com")
            .enable_citations(true)
            .build();

        let json = serde_json::to_string(&tool).unwrap();
        let parsed: WebFetchTool = serde_json::from_str(&json).unwrap();

        assert_eq!(tool.config(), parsed.config());
        assert_eq!(tool.kind(), parsed.kind());
    }

    #[test]
    fn test_web_fetch_tool_is_supported_by() {
        assert!(WebFetchTool::is_supported_by("anthropic"));
        assert!(WebFetchTool::is_supported_by("Anthropic"));
        assert!(WebFetchTool::is_supported_by("claude"));
        assert!(WebFetchTool::is_supported_by("google"));
        assert!(WebFetchTool::is_supported_by("Gemini"));

        assert!(!WebFetchTool::is_supported_by("openai"));
        assert!(!WebFetchTool::is_supported_by("cohere"));
    }

    #[test]
    fn test_web_fetch_error_display() {
        let err = WebFetchError::ConflictingDomainFilters;
        assert!(err.to_string().contains("Cannot specify both"));

        let err = WebFetchError::InvalidDomain("bad domain".to_string());
        assert!(err.to_string().contains("Invalid domain"));

        let err = WebFetchError::DomainNotAllowed("evil.com".to_string());
        assert!(err.to_string().contains("not in the allowed list"));

        let err = WebFetchError::DomainBlocked("evil.com".to_string());
        assert!(err.to_string().contains("blocked"));

        let err = WebFetchError::MaxUsesExceeded { current: 5, max: 5 };
        assert!(err.to_string().contains("5 of 5"));
    }

    #[test]
    fn test_web_fetch_builder_try_build() {
        // Valid
        let result = WebFetchTool::builder()
            .allow_domain("example.com")
            .try_build();
        assert!(result.is_ok());

        // The builder prevents conflicting states, so this should succeed
        let result = WebFetchTool::builder()
            .block_domain("evil.com")
            .try_build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_web_fetch_tool_equality() {
        let tool1 = WebFetchTool::builder()
            .max_uses(5)
            .allow_domain("example.com")
            .build();

        let tool2 = WebFetchTool::builder()
            .max_uses(5)
            .allow_domain("example.com")
            .build();

        let tool3 = WebFetchTool::builder()
            .max_uses(10)
            .allow_domain("example.com")
            .build();

        assert_eq!(tool1, tool2);
        assert_ne!(tool1, tool3);
    }
}
