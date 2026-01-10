//! UI protocol adapters for serdesAI.
//!
//! This crate provides adapters for integrating serdesAI agents with
//! frontend UI frameworks and streaming protocols:
//!
//! - **[`vercel_ai`]**: Vercel AI SDK Data Stream Protocol (SSE)
//! - **[`ag_ui`]**: AG-UI protocol for rich agent interactions
//!
//! # Feature Flags
//!
//! - `vercel` (default): Enable Vercel AI SDK adapter
//! - `ag-ui`: Enable AG-UI protocol adapter
//! - `full`: Enable all adapters
//!
//! # Example: Vercel AI SDK
//!
//! ```ignore
//! use serdes_ai_ui::vercel_ai::{
//!     VercelAIEventStream, Chunk, VERCEL_AI_DSP_HEADERS, chunks_to_sse
//! };
//! use serdes_ai_streaming::AgentStreamEvent;
//!
//! async fn handle_stream(agent_stream: impl Stream<Item = AgentStreamEvent<()>>) {
//!     let mut transformer = VercelAIEventStream::new();
//!
//!     // Emit SSE for start chunks
//!     let start_sse = chunks_to_sse(&transformer.before_stream());
//!     send_sse(&start_sse);
//!
//!     // Transform agent events to SSE
//!     while let Some(event) = agent_stream.next().await {
//!         let chunks = transformer.transform_event(event);
//!         send_sse(&chunks_to_sse(&chunks));
//!     }
//!
//!     // Emit SSE for end chunks
//!     let end_sse = chunks_to_sse(&transformer.after_stream());
//!     send_sse(&end_sse);
//! }
//! ```
//!
//! # Vercel AI Chunk Types
//!
//! The Vercel AI adapter provides these chunk types:
//!
//! | Category | Chunks |
//! |----------|--------|
//! | Lifecycle | `StartChunk`, `StartStepChunk`, `FinishStepChunk`, `FinishChunk`, `DoneChunk`, `AbortChunk` |
//! | Text | `TextStartChunk`, `TextDeltaChunk`, `TextEndChunk` |
//! | Reasoning | `ReasoningStartChunk`, `ReasoningDeltaChunk`, `ReasoningEndChunk` |
//! | Tool Input | `ToolInputStartChunk`, `ToolInputDeltaChunk`, `ToolInputAvailableChunk`, `ToolInputErrorChunk` |
//! | Tool Output | `ToolOutputAvailableChunk`, `ToolOutputErrorChunk`, `ToolOutputDeniedChunk` |
//! | Sources | `SourceUrlChunk`, `SourceDocumentChunk` |
//! | Data | `DataChunk`, `FileChunk` |
//! | Error | `ErrorChunk`, `MessageMetadataChunk` |

#![warn(missing_docs)]
#![deny(unsafe_code)]

#[cfg(feature = "vercel")]
pub mod vercel_ai;

#[cfg(feature = "ag-ui")]
pub mod ag_ui;

// Re-export commonly used types when features are enabled
#[cfg(feature = "vercel")]
pub use vercel_ai::{Chunk, FinishReason, VercelAIEventStream, VERCEL_AI_DSP_HEADERS};
