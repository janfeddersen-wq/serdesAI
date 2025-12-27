//! Vercel AI SDK adapter for serdesAI.
//!
//! This module provides streaming adapters compatible with the
//! [Vercel AI SDK](https://sdk.vercel.ai/) Data Stream Protocol.
//!
//! The Vercel AI SDK uses Server-Sent Events (SSE) with a specific
//! data format for streaming text, tool calls, reasoning, and other events.
//!
//! # Overview
//!
//! The main components are:
//!
//! - [`VercelAIEventStream`]: Transforms [`AgentStreamEvent`](serdes_ai_streaming::AgentStreamEvent)s
//!   into Vercel AI protocol chunks
//! - [`Chunk`]: Trait for all chunk types that can be serialized to SSE
//! - Various chunk types (`TextDeltaChunk`, `ToolInputStartChunk`, etc.)
//!
//! # Example
//!
//! ```ignore
//! use serdes_ai_ui::vercel_ai::{VercelAIEventStream, Chunk, VERCEL_AI_DSP_HEADERS};
//! use serdes_ai_streaming::AgentStreamEvent;
//!
//! // Create transformer
//! let mut transformer = VercelAIEventStream::new();
//!
//! // Set response headers
//! for (key, value) in VERCEL_AI_DSP_HEADERS {
//!     response.header(key, value);
//! }
//!
//! // Emit start chunks
//! for chunk in transformer.before_stream() {
//!     response.write(format!("data: {}\n\n", chunk.encode()));
//! }
//!
//! // Transform and emit agent events
//! while let Some(event) = agent_stream.next().await {
//!     for chunk in transformer.transform_event(event) {
//!         response.write(format!("data: {}\n\n", chunk.encode()));
//!     }
//! }
//!
//! // Emit end chunks
//! for chunk in transformer.after_stream() {
//!     response.write(format!("data: {}\n\n", chunk.encode()));
//! }
//! ```
//!
//! # Protocol Details
//!
//! The Vercel AI Data Stream Protocol sends JSON objects as SSE data lines.
//! Each object has a `type` field indicating the chunk type. The protocol
//! supports:
//!
//! - **Message lifecycle**: `start`, `start-step`, `finish-step`, `finish`, `done`
//! - **Text streaming**: `text-start`, `text-delta`, `text-end`
//! - **Reasoning**: `reasoning-start`, `reasoning-delta`, `reasoning-end`
//! - **Tool calls**: `tool-input-start`, `tool-input-delta`, `tool-input-available`
//! - **Tool results**: `tool-output-available`, `tool-output-error`
//! - **Errors**: `error`

mod stream;
mod types;

pub use stream::{
    chunks_to_sse, VercelAIEventStream, VERCEL_AI_DSP_HEADERS,
};
pub use types::{
    // Core trait and helper
    encode_chunk,
    Chunk,
    // Enums
    FinishReason,
    ProviderMetadata,
    UsageInfo,
    // Message lifecycle
    AbortChunk,
    DoneChunk,
    FinishChunk,
    FinishStepChunk,
    StartChunk,
    StartStepChunk,
    // Text streaming
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    // Reasoning/thinking
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    // Tool input
    ToolApprovalRequestChunk,
    ToolInputAvailableChunk,
    ToolInputDeltaChunk,
    ToolInputErrorChunk,
    ToolInputStartChunk,
    // Tool output
    ToolOutputAvailableChunk,
    ToolOutputDeniedChunk,
    ToolOutputErrorChunk,
    // Sources/citations
    SourceDocumentChunk,
    SourceUrlChunk,
    // Data
    DataChunk,
    FileChunk,
    // Error & metadata
    ErrorChunk,
    MessageMetadataChunk,
};
