//! # serdes-ai-streaming
//!
//! Streaming support for serdes-ai.
//!
//! This crate provides infrastructure for handling streaming responses
//! from LLM providers, including SSE parsing, async stream abstractions,
//! delta accumulation, and agent streaming.
//!
//! ## Core Concepts
//!
//! - **[`AgentStream`]**: Stream agent execution with typed events
//! - **[`AgentStreamEvent`]**: Events emitted during streaming (text, tools, thinking)
//! - **[`PartialResponse`]**: Accumulate deltas into complete responses
//! - **[`SseParser`]**: Parse Server-Sent Events from HTTP responses
//! - **Debouncing**: Temporal grouping for efficient streaming
//!
//! ## Example - Basic Streaming
//!
//! ```ignore
//! use serdes_ai_streaming::{AgentStream, AgentStreamExt};
//! use futures::StreamExt;
//!
//! let stream = AgentStream::new(delta_stream, "run-123");
//!
//! // Stream only text deltas
//! let mut text_stream = stream.text_deltas();
//! while let Some(text) = text_stream.next().await {
//!     print!("{}", text);
//! }
//! ```
//!
//! ## Example - SSE Parsing
//!
//! ```ignore
//! use serdes_ai_streaming::{SseParser, SseEventExt};
//!
//! let mut parser = SseParser::new();
//! parser.feed_str("data: {\"content\": \"hello\"}\n\n");
//!
//! while let Some(event) = parser.next_event() {
//!     if let Some(delta) = event.to_response_delta() {
//!         println!("Delta: {:?}", delta);
//!     }
//! }
//! ```
//!
//! ## Example - Debouncing
//!
//! ```ignore
//! use serdes_ai_streaming::{StreamDebounceExt, TextStreamExt};
//! use std::time::Duration;
//!
//! let debounced = text_stream
//!     .debounce(Duration::from_millis(50))
//!     .flat_map(|batch| futures::stream::iter(batch));
//!
//! // Or coalesce into larger chunks
//! let chunked = text_stream.coalesce(100, 1000);
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod agent_stream;
pub mod debounce;
pub mod error;
pub mod events;
pub mod partial_response;
pub mod parts_manager;
pub mod sse;

#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub mod websocket;

// Re-exports
pub use agent_stream::{
    AgentStream, AgentStreamExt, OutputStream, ResponseStream, StreamConfig, StreamState,
    TextDelta, TextDeltaStream,
};
pub use debounce::{
    CoalescedTextStream, DebouncedStream, StreamDebounceExt, TextStreamExt, ThrottledStream,
};
pub use error::{StreamError, StreamResult};
pub use events::AgentStreamEvent;
pub use partial_response::{PartialResponse, ResponseDelta};
pub use parts_manager::{ManagedPart, ModelResponsePartsManager, ToolCallAccumulator, VendorId};
pub use sse::{SseEvent, SseEventExt, SseParser, SseStream};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        AgentStream, AgentStreamEvent, AgentStreamExt, PartialResponse, ResponseDelta, SseEvent,
        SseEventExt, SseParser, StreamConfig, StreamDebounceExt, StreamError, StreamResult,
        StreamState, TextDelta, TextStreamExt,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let _ = StreamState::Pending;
        let config = StreamConfig::default();
        assert!(config.emit_partial_outputs);
    }
}
