//! Vercel AI event stream adapter.
//!
//! This module provides the [`VercelAIEventStream`] transformer that converts
//! serdesAI agent stream events to the Vercel AI Data Stream Protocol format.

use super::types::{self, *};
use serde_json::Value;
use serdes_ai_streaming::AgentStreamEvent;
use std::collections::HashMap;

/// HTTP headers for Vercel AI Data Stream Protocol responses.
pub const VERCEL_AI_DSP_HEADERS: &[(&str, &str)] = &[
    ("x-vercel-ai-ui-message-stream", "v1"),
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache"),
    ("connection", "keep-alive"),
];

/// State tracking for text streaming.
#[derive(Debug, Default)]
struct TextState {
    /// Whether we've emitted TextStartChunk.
    started: bool,
    /// Index of the current text part.
    part_index: Option<usize>,
}

/// State tracking for reasoning/thinking streaming.
#[derive(Debug, Default)]
struct ReasoningState {
    /// Whether we've emitted ReasoningStartChunk.
    started: bool,
    /// Index of the current reasoning part.
    index: Option<usize>,
}

/// State tracking for a tool call.
#[derive(Debug, Clone)]
struct ToolCallState {
    /// Tool call ID.
    tool_call_id: String,
    /// Tool name (kept for potential future use in multi-step).
    #[allow(dead_code)]
    tool_name: String,
    /// Accumulated arguments buffer.
    args_buffer: String,
    /// Whether we've emitted ToolInputStartChunk (for tracking).
    #[allow(dead_code)]
    started: bool,
}

/// Vercel AI event stream transformer.
///
/// Converts [`AgentStreamEvent`]s into Vercel AI Data Stream Protocol chunks.
/// Handles proper sequencing of start/delta/end events and maintains state
/// across the stream lifecycle.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_ui::vercel_ai::{VercelAIEventStream, Chunk};
/// use serdes_ai_streaming::AgentStreamEvent;
///
/// let mut transformer = VercelAIEventStream::new();
///
/// // Generate start chunks
/// let start_chunks = transformer.before_stream();
///
/// // Transform agent events
/// for event in agent_events {
///     let chunks = transformer.transform_event(event);
///     for chunk in chunks {
///         send_sse(chunk.encode());
///     }
/// }
///
/// // Generate end chunks
/// let end_chunks = transformer.after_stream();
/// ```
pub struct VercelAIEventStream {
    /// Counter for generating unique message IDs.
    message_id_counter: u32,
    /// Current message ID (set on stream start).
    current_message_id: Option<String>,
    /// Whether we've started a step.
    step_started: bool,
    /// Current step number.
    current_step: u32,
    /// Finish reason (set when stream completes).
    finish_reason: Option<FinishReason>,
    /// Text streaming state.
    text_state: TextState,
    /// Reasoning streaming state.
    reasoning_state: ReasoningState,
    /// Tool call states by index.
    tool_calls: HashMap<usize, ToolCallState>,
    /// Whether we have pending tool calls.
    has_pending_tool_calls: bool,
    /// Usage information.
    usage: Option<UsageInfo>,
}

impl Default for VercelAIEventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl VercelAIEventStream {
    /// Create a new Vercel AI event stream transformer.
    pub fn new() -> Self {
        Self {
            message_id_counter: 0,
            current_message_id: None,
            step_started: false,
            current_step: 0,
            finish_reason: None,
            text_state: TextState::default(),
            reasoning_state: ReasoningState::default(),
            tool_calls: HashMap::new(),
            has_pending_tool_calls: false,
            usage: None,
        }
    }

    /// Generate a new unique message ID.
    pub fn new_message_id(&mut self) -> String {
        self.message_id_counter += 1;
        format!("msg-{}", self.message_id_counter)
    }

    /// Get the current message ID, generating one if needed.
    pub fn current_message_id(&mut self) -> String {
        if let Some(ref id) = self.current_message_id {
            id.clone()
        } else {
            let id = self.new_message_id();
            self.current_message_id = Some(id.clone());
            id
        }
    }

    /// Encode a chunk for SSE transmission.
    ///
    /// Formats the chunk as an SSE data line.
    pub fn encode_event<C: ?Sized + erased_serde::Serialize>(event: &C) -> String {
        format!("data: {}\n\n", types::encode_chunk(event))
    }

    /// Generate chunks to emit before the stream starts.
    ///
    /// This emits:
    /// - `StartChunk` to begin the message
    /// - `StartStepChunk` to begin the first step
    pub fn before_stream(&mut self) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        let message_id = self.current_message_id();
        chunks.push(Box::new(StartChunk::new(&message_id)));
        chunks.push(Box::new(StartStepChunk::new(&message_id).with_step(0)));

        self.step_started = true;
        self.current_step = 0;

        chunks
    }

    /// Generate chunks to emit after the stream ends.
    ///
    /// This emits:
    /// - `TextEndChunk` if text was being streamed
    /// - `ReasoningEndChunk` if reasoning was being streamed
    /// - `FinishStepChunk` for the current step
    /// - `FinishChunk` for the message
    /// - `DoneChunk` to signal stream completion
    pub fn after_stream(&mut self) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();
        let message_id = self.current_message_id();

        // Close any open text stream
        if self.text_state.started {
            chunks.push(Box::new(TextEndChunk::new()));
            self.text_state.started = false;
        }

        // Close any open reasoning stream
        if self.reasoning_state.started {
            chunks.push(Box::new(ReasoningEndChunk::new()));
            self.reasoning_state.started = false;
        }

        let finish_reason = self.finish_reason.unwrap_or(FinishReason::Stop);

        // Finish the current step
        if self.step_started {
            let mut finish_step = FinishStepChunk::new(&message_id, finish_reason);
            if let Some(ref usage) = self.usage {
                finish_step = finish_step.with_usage(usage.clone());
            }
            if self.has_pending_tool_calls {
                finish_step = finish_step.with_continued(true);
            }
            chunks.push(Box::new(finish_step));
            self.step_started = false;
        }

        // Finish the message
        let mut finish = FinishChunk::new(&message_id, finish_reason);
        if let Some(ref usage) = self.usage {
            finish = finish.with_usage(usage.clone());
        }
        chunks.push(Box::new(finish));

        // Done!
        chunks.push(Box::new(DoneChunk::new()));

        chunks
    }

    /// Transform an agent stream event into Vercel AI chunks.
    ///
    /// This is the core transformation logic that maps serdesAI events
    /// to the Vercel AI protocol.
    pub fn transform_event<O>(&mut self, event: AgentStreamEvent<O>) -> Vec<Box<dyn Chunk>> {
        match event {
            AgentStreamEvent::RunStart { .. } => {
                // RunStart is handled by before_stream()
                vec![]
            }

            AgentStreamEvent::RequestStart { step } => self.handle_request_start(step),

            AgentStreamEvent::TextDelta {
                content,
                part_index,
            } => self.handle_text_delta(content, part_index),

            AgentStreamEvent::ThinkingDelta { content, index } => {
                self.handle_thinking_delta(content, index)
            }

            AgentStreamEvent::ToolCallStart {
                name,
                tool_call_id,
                index,
            } => self.handle_tool_call_start(name, tool_call_id, index),

            AgentStreamEvent::ToolCallDelta { args_delta, index } => {
                self.handle_tool_call_delta(args_delta, index)
            }

            AgentStreamEvent::ToolCallComplete { name, args, index } => {
                self.handle_tool_call_complete(name, args, index)
            }

            AgentStreamEvent::ToolResult {
                name: _,
                result,
                success,
                index,
            } => self.handle_tool_result(result, success, index),

            AgentStreamEvent::UsageUpdate { usage } => self.handle_usage_update(usage),

            AgentStreamEvent::ResponseComplete { .. } => {
                // Response complete is informational, actual finish in RunComplete
                vec![]
            }

            AgentStreamEvent::PartialOutput { .. } | AgentStreamEvent::FinalOutput { .. } => {
                // Output events are for typed output, not UI streaming
                vec![]
            }

            AgentStreamEvent::RunComplete { .. } => {
                self.finish_reason = Some(FinishReason::Stop);
                // RunComplete is handled by after_stream()
                vec![]
            }

            AgentStreamEvent::Error { message, .. } => self.handle_error(message),
        }
    }

    /// Handle a new request/step starting.
    fn handle_request_start(&mut self, step: u32) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();
        let message_id = self.current_message_id();

        // If we already started a step, finish it first
        if self.step_started && step > self.current_step {
            // Close text/reasoning if open
            if self.text_state.started {
                chunks.push(Box::new(TextEndChunk::new()));
                self.text_state.started = false;
            }
            if self.reasoning_state.started {
                chunks.push(Box::new(ReasoningEndChunk::new()));
                self.reasoning_state.started = false;
            }

            // Finish the previous step
            let finish_step =
                FinishStepChunk::new(&message_id, FinishReason::ToolCalls).with_continued(true);
            chunks.push(Box::new(finish_step));

            // Start the new step
            chunks.push(Box::new(StartStepChunk::new(&message_id).with_step(step)));
            self.current_step = step;
        }

        chunks
    }

    /// Handle a text delta event.
    fn handle_text_delta(&mut self, content: String, part_index: usize) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        // Close reasoning if we were in reasoning mode
        if self.reasoning_state.started {
            chunks.push(Box::new(ReasoningEndChunk::new()));
            self.reasoning_state.started = false;
        }

        // Check if we need to start text or if part index changed
        let need_start = !self.text_state.started || self.text_state.part_index != Some(part_index);

        if need_start {
            // Close previous text part if needed
            if self.text_state.started {
                chunks.push(Box::new(TextEndChunk::new()));
            }

            chunks.push(Box::new(TextStartChunk::new()));
            self.text_state.started = true;
            self.text_state.part_index = Some(part_index);
        }

        chunks.push(Box::new(TextDeltaChunk::new(content)));
        chunks
    }

    /// Handle a thinking/reasoning delta event.
    fn handle_thinking_delta(&mut self, content: String, index: usize) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        // Close text if we were in text mode
        if self.text_state.started {
            chunks.push(Box::new(TextEndChunk::new()));
            self.text_state.started = false;
        }

        // Check if we need to start reasoning or if index changed
        let need_start = !self.reasoning_state.started || self.reasoning_state.index != Some(index);

        if need_start {
            // Close previous reasoning if needed
            if self.reasoning_state.started {
                chunks.push(Box::new(ReasoningEndChunk::new()));
            }

            chunks.push(Box::new(ReasoningStartChunk::new()));
            self.reasoning_state.started = true;
            self.reasoning_state.index = Some(index);
        }

        chunks.push(Box::new(ReasoningDeltaChunk::new(content)));
        chunks
    }

    /// Handle tool call start.
    fn handle_tool_call_start(
        &mut self,
        name: String,
        tool_call_id: Option<String>,
        index: usize,
    ) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        // Close text/reasoning if open
        if self.text_state.started {
            chunks.push(Box::new(TextEndChunk::new()));
            self.text_state.started = false;
        }
        if self.reasoning_state.started {
            chunks.push(Box::new(ReasoningEndChunk::new()));
            self.reasoning_state.started = false;
        }

        // Generate tool call ID if not provided
        let call_id = tool_call_id.unwrap_or_else(|| format!("call-{}", index));

        // Store state
        self.tool_calls.insert(
            index,
            ToolCallState {
                tool_call_id: call_id.clone(),
                tool_name: name.clone(),
                args_buffer: String::new(),
                started: true,
            },
        );

        self.has_pending_tool_calls = true;

        chunks.push(Box::new(ToolInputStartChunk::new(&call_id, &name)));
        chunks
    }

    /// Handle tool call arguments delta.
    fn handle_tool_call_delta(&mut self, args_delta: String, index: usize) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        if let Some(state) = self.tool_calls.get_mut(&index) {
            // Accumulate args
            state.args_buffer.push_str(&args_delta);

            // Emit delta chunk
            chunks.push(Box::new(ToolInputDeltaChunk::new(
                &state.tool_call_id,
                args_delta,
            )));
        }

        chunks
    }

    /// Handle tool call complete.
    fn handle_tool_call_complete(
        &mut self,
        name: String,
        args: Value,
        index: usize,
    ) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        // Get or create tool call state
        let tool_call_id = if let Some(state) = self.tool_calls.get(&index) {
            state.tool_call_id.clone()
        } else {
            // Tool call started without ToolCallStart event
            let call_id = format!("call-{}", index);
            self.tool_calls.insert(
                index,
                ToolCallState {
                    tool_call_id: call_id.clone(),
                    tool_name: name.clone(),
                    args_buffer: String::new(),
                    started: false,
                },
            );
            call_id
        };

        self.finish_reason = Some(FinishReason::ToolCalls);
        self.has_pending_tool_calls = true;

        chunks.push(Box::new(ToolInputAvailableChunk::new(
            &tool_call_id,
            &name,
            args,
        )));

        chunks
    }

    /// Handle tool result.
    fn handle_tool_result(
        &mut self,
        result: Value,
        success: bool,
        index: usize,
    ) -> Vec<Box<dyn Chunk>> {
        let mut chunks: Vec<Box<dyn Chunk>> = Vec::new();

        // Get tool call ID
        let tool_call_id = self
            .tool_calls
            .get(&index)
            .map(|s| s.tool_call_id.clone())
            .unwrap_or_else(|| format!("call-{}", index));

        if success {
            chunks.push(Box::new(ToolOutputAvailableChunk::new(
                &tool_call_id,
                result,
            )));
        } else {
            let error_msg = result
                .as_str()
                .unwrap_or("Tool execution failed")
                .to_string();
            chunks.push(Box::new(ToolOutputErrorChunk::new(
                &tool_call_id,
                error_msg,
            )));
        }

        // Tool call is resolved
        self.tool_calls.remove(&index);
        if self.tool_calls.is_empty() {
            self.has_pending_tool_calls = false;
        }

        chunks
    }

    /// Handle usage update.
    fn handle_usage_update(&mut self, usage: serdes_ai_core::RequestUsage) -> Vec<Box<dyn Chunk>> {
        self.usage = Some(
            UsageInfo::new()
                .with_prompt_tokens(usage.request_tokens.unwrap_or(0) as u32)
                .with_completion_tokens(usage.response_tokens.unwrap_or(0) as u32)
                .with_total_tokens(usage.total_tokens.unwrap_or(0) as u32),
        );
        vec![]
    }

    /// Handle error event.
    fn handle_error(&mut self, message: String) -> Vec<Box<dyn Chunk>> {
        self.finish_reason = Some(FinishReason::Error);
        vec![Box::new(ErrorChunk::new(message))]
    }
}

/// Helper to create SSE response body from chunks.
///
/// Concatenates all chunks as SSE data lines.
pub fn chunks_to_sse(chunks: &[Box<dyn Chunk>]) -> String {
    chunks
        .iter()
        .map(|c| VercelAIEventStream::encode_event(&**c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_event_stream() {
        let stream = VercelAIEventStream::new();
        assert!(stream.current_message_id.is_none());
        assert!(!stream.step_started);
    }

    #[test]
    fn test_message_id_generation() {
        let mut stream = VercelAIEventStream::new();
        let id1 = stream.new_message_id();
        let id2 = stream.new_message_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("msg-"));
    }

    #[test]
    fn test_before_stream() {
        let mut stream = VercelAIEventStream::new();
        let chunks = stream.before_stream();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_type(), "start");
        assert_eq!(chunks[1].chunk_type(), "start-step");
        assert!(stream.step_started);
    }

    #[test]
    fn test_after_stream() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        let chunks = stream.after_stream();

        // Should have finish-step, finish, done
        assert!(chunks.iter().any(|c| c.chunk_type() == "finish-step"));
        assert!(chunks.iter().any(|c| c.chunk_type() == "finish"));
        assert!(chunks.iter().any(|c| c.chunk_type() == "done"));
    }

    #[test]
    fn test_text_delta_transformation() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        // First text delta should emit text-start + text-delta
        let event: AgentStreamEvent<()> = AgentStreamEvent::TextDelta {
            content: "Hello".to_string(),
            part_index: 0,
        };
        let chunks = stream.transform_event(event);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_type(), "text-start");
        assert_eq!(chunks[1].chunk_type(), "text-delta");

        // Second text delta should only emit text-delta
        let event2: AgentStreamEvent<()> = AgentStreamEvent::TextDelta {
            content: " World".to_string(),
            part_index: 0,
        };
        let chunks2 = stream.transform_event(event2);

        assert_eq!(chunks2.len(), 1);
        assert_eq!(chunks2[0].chunk_type(), "text-delta");
    }

    #[test]
    fn test_thinking_delta_transformation() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        let event: AgentStreamEvent<()> = AgentStreamEvent::ThinkingDelta {
            content: "Let me think...".to_string(),
            index: 0,
        };
        let chunks = stream.transform_event(event);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_type(), "reasoning-start");
        assert_eq!(chunks[1].chunk_type(), "reasoning-delta");
    }

    #[test]
    fn test_tool_call_transformation() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        // Tool call start
        let event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallStart {
            name: "get_weather".to_string(),
            tool_call_id: Some("call-123".to_string()),
            index: 0,
        };
        let chunks = stream.transform_event(event);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type(), "tool-input-start");

        // Tool call delta
        let delta_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallDelta {
            args_delta: "{\"city\":".to_string(),
            index: 0,
        };
        let delta_chunks = stream.transform_event(delta_event);

        assert_eq!(delta_chunks.len(), 1);
        assert_eq!(delta_chunks[0].chunk_type(), "tool-input-delta");

        // Tool call complete
        let complete_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallComplete {
            name: "get_weather".to_string(),
            args: serde_json::json!({"city": "London"}),
            index: 0,
        };
        let complete_chunks = stream.transform_event(complete_event);

        assert_eq!(complete_chunks.len(), 1);
        assert_eq!(complete_chunks[0].chunk_type(), "tool-input-available");
    }

    #[test]
    fn test_tool_result_success() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        // First start a tool call
        let start_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallStart {
            name: "get_weather".to_string(),
            tool_call_id: Some("call-123".to_string()),
            index: 0,
        };
        stream.transform_event(start_event);

        // Then send result
        let event: AgentStreamEvent<()> = AgentStreamEvent::ToolResult {
            name: "get_weather".to_string(),
            result: serde_json::json!({"temp": 20}),
            success: true,
            index: 0,
        };
        let chunks = stream.transform_event(event);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type(), "tool-output-available");
    }

    #[test]
    fn test_tool_result_error() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        // First start a tool call
        let start_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallStart {
            name: "get_weather".to_string(),
            tool_call_id: Some("call-123".to_string()),
            index: 0,
        };
        stream.transform_event(start_event);

        // Then send error result
        let event: AgentStreamEvent<()> = AgentStreamEvent::ToolResult {
            name: "get_weather".to_string(),
            result: serde_json::json!("City not found"),
            success: false,
            index: 0,
        };
        let chunks = stream.transform_event(event);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type(), "tool-output-error");
    }

    #[test]
    fn test_error_transformation() {
        let mut stream = VercelAIEventStream::new();
        stream.before_stream();

        let event: AgentStreamEvent<()> = AgentStreamEvent::Error {
            message: "Something went wrong".to_string(),
            recoverable: false,
        };
        let chunks = stream.transform_event(event);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type(), "error");
        assert_eq!(stream.finish_reason, Some(FinishReason::Error));
    }

    #[test]
    fn test_encode_event() {
        let chunk = TextDeltaChunk::new("Hello");
        let encoded = VercelAIEventStream::encode_event(&chunk);

        assert!(encoded.starts_with("data: "));
        assert!(encoded.ends_with("\n\n"));
        assert!(encoded.contains("text-delta"));
    }

    #[test]
    fn test_chunks_to_sse() {
        let chunks: Vec<Box<dyn Chunk>> = vec![
            Box::new(TextStartChunk::new()),
            Box::new(TextDeltaChunk::new("Hi")),
        ];

        let sse = chunks_to_sse(&chunks);
        assert!(sse.contains("text-start"));
        assert!(sse.contains("text-delta"));
        assert_eq!(sse.matches("data: ").count(), 2);
    }

    #[test]
    fn test_full_stream_lifecycle() {
        let mut stream = VercelAIEventStream::new();

        // Before
        let before = stream.before_stream();
        assert!(!before.is_empty());

        // Some events
        let text: AgentStreamEvent<()> = AgentStreamEvent::text_delta("Hello", 0);
        let text_chunks = stream.transform_event(text);
        assert!(!text_chunks.is_empty());

        // After
        let after = stream.after_stream();
        assert!(after.iter().any(|c| c.chunk_type() == "done"));
    }
}
