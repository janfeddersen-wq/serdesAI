//! Groq HTTP client utilities.

use reqwest::{Client, ClientBuilder};
use std::time::Duration;

/// Create a Groq API client with default settings.
pub fn create_client() -> Client {
    ClientBuilder::new()
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
}

/// Groq API error response.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ErrorResponse {
    /// Error details.
    pub error: ErrorDetail,
}

/// Error detail.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ErrorDetail {
    /// Error message.
    pub message: String,
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    /// Error code.
    pub code: Option<String>,
}

impl std::fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error.message)
    }
}

/// Parse an error response from Groq.
pub fn parse_error(status: u16, body: &str) -> String {
    if let Ok(err) = serde_json::from_str::<ErrorResponse>(body) {
        err.error.message
    } else {
        format!("HTTP {}: {}", status, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error() {
        let body = r#"{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}"#;
        let msg = parse_error(429, body);
        assert_eq!(msg, "Rate limit exceeded");
    }

    #[test]
    fn test_parse_error_invalid_json() {
        let msg = parse_error(500, "Internal server error");
        assert!(msg.contains("500"));
    }
}
