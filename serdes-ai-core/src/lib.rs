//! # serdes-ai-core
//!
//! Core types, messages, and error handling for the serdes-ai framework.
//!
//! This crate provides the foundational types used throughout the serdes-ai ecosystem:
//!
//! - **Messages**: Request/response message types for LLM interactions
//! - **Errors**: Comprehensive error types with context
//! - **Usage**: Token usage tracking and limits
//! - **Settings**: Model configuration options
//! - **Identifiers**: Type-safe IDs for conversations, messages, runs
//!
//! ## Feature Flags
//!
//! - `tracing-integration`: Enable tracing instrumentation
//! - `otel`: Enable OpenTelemetry integration
//! - `full`: Enable all optional features
//!
//! ## Example
//!
//! ```rust
//! use serdes_ai_core::{
//!     messages::{ModelRequest, ModelResponse, UserContent},
//!     usage::{RequestUsage, RunUsage, UsageLimits},
//!     settings::ModelSettings,
//!     identifier::{generate_run_id, now_utc},
//! };
//!
//! // Build a request
//! let mut request = ModelRequest::new();
//! request.add_system_prompt("You are a helpful assistant.");
//! request.add_user_prompt("Hello!");
//!
//! // Configure settings
//! let settings = ModelSettings::new()
//!     .max_tokens(1000)
//!     .temperature(0.7);
//!
//! // Track usage
//! let mut usage = RunUsage::new();
//! usage.add_request(RequestUsage::with_tokens(100, 50));
//!
//! // Check limits
//! let limits = UsageLimits::new().max_total_tokens(10000);
//! limits.check(&usage).expect("Within limits");
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod errors;
pub mod format;
pub mod identifier;
pub mod messages;
pub mod settings;
pub mod usage;

// Re-exports for convenience
pub use errors::{Result, SerdesAiError};
pub use format::{format_as_xml, format_as_xml_with_options, XmlFormatError, XmlFormatOptions};
pub use identifier::{now_utc, ConversationId, RunId, ToolCallId};
pub use messages::{
    BinaryContent,
    // Builtin tools (web search, code execution, file search)
    BuiltinToolCallPart,
    BuiltinToolReturnContent,
    BuiltinToolReturnPart,
    CodeExecutionResult,
    // File and binary content
    FilePart,
    FileSearchResult,
    FileSearchResults,
    FinishReason,
    // Core request/response
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponsePartDelta,
    // Streaming events
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    // Text and thinking
    TextPart,
    ThinkingPart,
    // Tool calls and returns
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserContentPart,
    UserPromptPart,
    WebSearchResult,
    WebSearchResults,
};
pub use settings::ModelSettings;
pub use usage::{RequestUsage, RunUsage, UsageLimits};

/// Prelude module for common imports.
///
/// ```rust
/// use serdes_ai_core::prelude::*;
/// ```
pub mod prelude {
    pub use crate::errors::{Result, SerdesAiError};
    pub use crate::format::{format_as_xml, format_as_xml_with_options, XmlFormatOptions};
    pub use crate::identifier::{
        generate_run_id, generate_tool_call_id, now_utc, ConversationId, RunId, ToolCallId,
    };
    pub use crate::messages::{
        BinaryContent,
        // Builtin tools
        BuiltinToolCallPart,
        BuiltinToolReturnContent,
        BuiltinToolReturnPart,
        CodeExecutionResult,
        // File and binary content
        FilePart,
        FileSearchResult,
        FileSearchResults,
        // Core request/response
        FinishReason,
        ModelRequest,
        ModelRequestPart,
        ModelResponse,
        ModelResponsePart,
        ModelResponsePartDelta,
        // Streaming
        ModelResponseStreamEvent,
        SystemPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallArgs,
        ToolCallPart,
        ToolReturnPart,
        UserContent,
        UserContentPart,
        UserPromptPart,
        WebSearchResult,
        WebSearchResults,
    };
    pub use crate::settings::ModelSettings;
    pub use crate::usage::{RequestUsage, RunUsage, UsageLimits};
}
