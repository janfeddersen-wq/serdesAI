//! AG-UI protocol adapter for serdesAI.
//!
//! This module provides adapters for the AG-UI (Agent-User Interaction) protocol,
//! enabling rich agent interactions with frontend applications.
//!
//! # Overview
//!
//! AG-UI is a real-time streaming protocol for agent-user communication.
//! It supports:
//!
//! - **Run lifecycle**: `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`
//! - **Text messages**: Streaming text with start/content/end events
//! - **Thinking**: Nested thinking blocks with their own text messages
//! - **Tool calls**: Start/args/end/result lifecycle
//! - **State management**: Snapshots and deltas
//!
//! # Example
//!
//! ```ignore
//! use serdes_ai_ui::ag_ui::{AgUiEventStream, OutputFormat, Event};
//! use serdes_ai_streaming::AgentStreamEvent;
//!
//! async fn handle_stream(agent_stream: impl Stream<Item = AgentStreamEvent<()>>) {
//!     let mut transformer = AgUiEventStream::new("thread-123", "run-456");
//!
//!     // Emit start events
//!     for event in transformer.before_stream() {
//!         send(AgUiEventStream::encode_event(&*event, OutputFormat::Ndjson));
//!     }
//!
//!     // Transform agent events
//!     while let Some(event) = agent_stream.next().await {
//!         for ev in transformer.transform_event(event) {
//!             send(AgUiEventStream::encode_event(&*ev, OutputFormat::Ndjson));
//!         }
//!     }
//!
//!     // Emit end events
//!     for event in transformer.after_stream() {
//!         send(AgUiEventStream::encode_event(&*event, OutputFormat::Ndjson));
//!     }
//! }
//! ```
//!
//! # Output Formats
//!
//! AG-UI supports two output formats:
//!
//! - **SSE**: Server-Sent Events (`data: {...}\n\n`)
//! - **NDJSON**: Newline-delimited JSON (`{...}\n`)

mod stream;
mod types;

pub use stream::{events_to_output, AgUiEventStream, OutputFormat};
pub use types::{
    // Core trait and helper
    encode_event,
    // Custom
    CustomEvent,
    Event,
    EventType,
    // State
    MessagesSnapshotEvent,
    RawEvent,
    // Run lifecycle
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    // Text messages
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    // Thinking
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    // Tool calls
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
};
