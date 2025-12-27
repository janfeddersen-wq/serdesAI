//! Common third-party tool integrations.
//!
//! This module provides ready-to-use tools that integrate with popular
//! third-party services for search, data retrieval, and more.
//!
//! ## Available Tools
//!
//! - **[`DuckDuckGoTool`]**: Web search using DuckDuckGo's Instant Answer API
//! - **[`TavilyTool`]**: AI-optimized web search using Tavily's API
//!
//! ## Feature Flag
//!
//! These tools require the `common-tools` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! serdes-ai-tools = { version = "0.1", features = ["common-tools"] }
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_tools::common::{DuckDuckGoTool, TavilyTool};
//!
//! // DuckDuckGo search (no API key required)
//! let ddg = DuckDuckGoTool::new();
//!
//! // Tavily search (requires API key)
//! let tavily = TavilyTool::new("your-api-key");
//! ```

mod duckduckgo;
mod tavily;

pub use duckduckgo::{DuckDuckGoConfig, DuckDuckGoResult, DuckDuckGoTool};
pub use tavily::{TavilyConfig, TavilyResult, TavilySearchDepth, TavilyTool};
