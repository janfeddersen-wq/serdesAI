//! Request formatting utilities.

use serde::Serialize;

/// Format a request body for logging (hiding sensitive data).
#[must_use]
pub fn format_request<T: Serialize>(request: &T) -> String {
    serde_json::to_string_pretty(request).unwrap_or_else(|_| "<serialization error>".to_string())
}
