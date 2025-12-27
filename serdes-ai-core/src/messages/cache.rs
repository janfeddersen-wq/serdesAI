//! Cache point type for caching prompts.
//!
//! This module defines the cache point marker that can be inserted into
//! message sequences to indicate caching boundaries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A cache point marker in the message sequence.
///
/// Cache points indicate where the prompt prefix can be cached by
/// providers that support prompt caching (e.g., Anthropic).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePoint {
    /// When this cache point was created.
    pub timestamp: DateTime<Utc>,
    /// Cache type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type: Option<CacheType>,
}

impl CachePoint {
    /// Part kind identifier.
    pub const PART_KIND: &'static str = "cache-point";

    /// Create a new cache point.
    #[must_use]
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            cache_type: None,
        }
    }

    /// Create a cache point at a specific timestamp.
    #[must_use]
    pub fn at(timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp,
            cache_type: None,
        }
    }

    /// Get the part kind.
    #[must_use]
    pub fn part_kind(&self) -> &'static str {
        Self::PART_KIND
    }

    /// Set the cache type.
    #[must_use]
    pub fn with_cache_type(mut self, cache_type: CacheType) -> Self {
        self.cache_type = Some(cache_type);
        self
    }

    /// Create an ephemeral cache point.
    #[must_use]
    pub fn ephemeral() -> Self {
        Self::new().with_cache_type(CacheType::Ephemeral)
    }
}

impl Default for CachePoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of caching to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheType {
    /// Ephemeral cache (short-lived).
    Ephemeral,
    /// Persistent cache (long-lived).
    Persistent,
}

impl Default for CacheType {
    fn default() -> Self {
        Self::Ephemeral
    }
}

impl std::fmt::Display for CacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ephemeral => write!(f, "ephemeral"),
            Self::Persistent => write!(f, "persistent"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_point_new() {
        let cp = CachePoint::new();
        assert_eq!(cp.part_kind(), "cache-point");
        assert!(cp.cache_type.is_none());
    }

    #[test]
    fn test_cache_point_ephemeral() {
        let cp = CachePoint::ephemeral();
        assert_eq!(cp.cache_type, Some(CacheType::Ephemeral));
    }

    #[test]
    fn test_serde_roundtrip() {
        let cp = CachePoint::ephemeral();
        let json = serde_json::to_string(&cp).unwrap();
        let parsed: CachePoint = serde_json::from_str(&json).unwrap();
        assert_eq!(cp.cache_type, parsed.cache_type);
    }
}
