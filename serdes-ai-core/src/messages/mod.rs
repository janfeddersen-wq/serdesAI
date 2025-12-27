//! Message types for model interactions.
//!
//! This module provides all the message types used for communicating with LLM models:
//!
//! - **Request types**: [`ModelRequest`], [`ModelRequestPart`], and related parts
//! - **Response types**: [`ModelResponse`], [`ModelResponsePart`], and related parts
//! - **Content types**: [`UserContent`] and multi-modal content parts
//! - **Tool types**: [`ToolCallPart`], [`ToolReturnPart`], and related
//! - **Streaming**: [`ModelResponseStreamEvent`] and delta types
//! - **Caching**: [`CachePoint`] for prompt caching
//!
//! ## Example
//!
//! ```rust
//! use serdes_ai_core::messages::{
//!     ModelRequest, ModelRequestPart, SystemPromptPart, UserPromptPart,
//!     ModelResponse, ModelResponsePart,
//! };
//!
//! // Build a request
//! let mut request = ModelRequest::new();
//! request.add_system_prompt("You are a helpful assistant.");
//! request.add_user_prompt("Hello!");
//!
//! // Process a response
//! let response = ModelResponse::text("Hello! How can I help you today?");
//! println!("Response: {}", response.text_content());
//! ```

pub mod cache;
pub mod content;
pub mod events;
pub mod media;
pub mod parts;
pub mod request;
pub mod response;
pub mod tool_return;

// Re-exports for convenience
pub use cache::{CachePoint, CacheType};
pub use content::{
    AudioContent, AudioUrl, BinaryAudio, BinaryDocument, BinaryFile, BinaryImage, BinaryVideo,
    DocumentContent, DocumentUrl, FileContent, FileUrl, ImageContent, ImageUrl, UserContent,
    UserContentPart, VideoContent, VideoUrl,
};
pub use events::{
    BuiltinToolCallPartDelta, ModelResponsePartDelta, ModelResponseStreamEvent, PartDeltaEvent,
    PartEndEvent, PartStartEvent, TextPartDelta, ThinkingPartDelta, ToolCallPartDelta,
};
pub use media::{AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType};
pub use parts::{
    BinaryContent, BuiltinToolCallPart, BuiltinToolReturnContent, BuiltinToolReturnPart,
    CodeExecutionResult, FilePart, FileSearchResult, FileSearchResults, TextPart, ThinkingPart,
    ToolCallArgs, ToolCallPart, WebSearchResult, WebSearchResults,
};
pub use request::{
    ModelRequest, ModelRequestPart, RetryContent, RetryPromptPart, SystemPromptPart,
    ToolReturnPart, UserPromptPart,
};
pub use response::{FinishReason, ModelResponse, ModelResponsePart};
pub use tool_return::{ToolReturn, ToolReturnContent, ToolReturnError, ToolReturnItem};
