//! AG-UI event stream adapter.
//!
//! This module provides the [`AgUiEventStream`] transformer that converts
//! serdesAI agent stream events to the AG-UI protocol format.

use super::types::{self, *};
use serdes_ai_streaming::AgentStreamEvent;
use std::collections::HashMap;

/// Output format for AG-UI events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Server-Sent Events format.
    #[default]
    Sse,
    /// Newline-delimited JSON format.
    Ndjson,
}

/// State tracking for text message streaming.
#[derive(Debug, Default)]
struct TextMessageState {
    /// Current message ID.
    message_id: Option<String>,
    /// Whether text message has started.
    started: bool,
    /// Part index being streamed.
    part_index: Option<usize>,
}

/// State tracking for thinking/reasoning.
#[derive(Debug, Default)]
struct ThinkingState {
    /// Whether thinking has started.
    started: bool,
    /// Whether thinking text message has started.
    text_started: bool,
    /// Current thinking index.
    index: Option<usize>,
}

/// State tracking for a tool call.
#[derive(Debug, Clone)]
struct ToolCallState {
    /// Tool call ID.
    tool_call_id: String,
    /// Tool name.
    #[allow(dead_code)]
    tool_name: String,
    /// Parent message ID (kept for potential multi-message tools).
    #[allow(dead_code)]
    parent_message_id: Option<String>,
    /// Whether args have started streaming.
    #[allow(dead_code)]
    args_started: bool,
}

/// AG-UI event stream transformer.
///
/// Converts [`AgentStreamEvent`]s into AG-UI protocol events.
/// Handles proper sequencing of lifecycle events and maintains state
/// across the stream.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_ui::ag_ui::{AgUiEventStream, OutputFormat};
/// use serdes_ai_streaming::AgentStreamEvent;
///
/// let mut transformer = AgUiEventStream::new("thread-123", "run-456");
///
/// // Generate start events
/// let start_events = transformer.before_stream();
///
/// // Transform agent events
/// for event in agent_events {
///     let ag_ui_events = transformer.transform_event(event);
///     for ev in ag_ui_events {
///         send(AgUiEventStream::encode_event(&*ev, OutputFormat::Sse));
///     }
/// }
///
/// // Generate end events
/// let end_events = transformer.after_stream();
/// ```
pub struct AgUiEventStream {
    /// Thread identifier.
    thread_id: String,
    /// Run identifier.
    run_id: String,
    /// Counter for generating unique message IDs.
    message_id_counter: u32,
    /// Current message ID.
    current_message_id: Option<String>,
    /// Text message state.
    text_state: TextMessageState,
    /// Thinking state.
    thinking_state: ThinkingState,
    /// Tool call states by index.
    tool_calls: HashMap<usize, ToolCallState>,
    /// Mapping from tool call ID to index.
    tool_call_id_to_index: HashMap<String, usize>,
    /// Whether we had an error.
    had_error: bool,
    /// Whether the stream has started.
    stream_started: bool,
}

impl AgUiEventStream {
    /// Create a new AG-UI event stream transformer.
    pub fn new(thread_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            run_id: run_id.into(),
            message_id_counter: 0,
            current_message_id: None,
            text_state: TextMessageState::default(),
            thinking_state: ThinkingState::default(),
            tool_calls: HashMap::new(),
            tool_call_id_to_index: HashMap::new(),
            had_error: false,
            stream_started: false,
        }
    }

    /// Get the current timestamp in milliseconds.
    pub fn get_timestamp() -> i64 {
        chrono::Utc::now().timestamp_millis()
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

    /// Encode an event for transmission.
    ///
    /// # Arguments
    /// * `event` - The event to encode
    /// * `format` - Output format (SSE or NDJSON)
    pub fn encode_event<E: ?Sized + erased_serde::Serialize>(
        event: &E,
        format: OutputFormat,
    ) -> String {
        let json = types::encode_event(event);
        match format {
            OutputFormat::Sse => format!("data: {}\n\n", json),
            OutputFormat::Ndjson => format!("{}\n", json),
        }
    }

    /// Generate events to emit before the stream starts.
    ///
    /// Emits `RunStartedEvent`.
    pub fn before_stream(&mut self) -> Vec<Box<dyn Event>> {
        self.stream_started = true;
        vec![Box::new(RunStartedEvent::new(
            &self.thread_id,
            &self.run_id,
        ))]
    }

    /// Generate events to emit after the stream ends.
    ///
    /// Closes any open messages/thinking and emits `RunFinishedEvent`.
    pub fn after_stream(&mut self) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        // Close thinking text if open
        if self.thinking_state.text_started {
            events.push(Box::new(ThinkingTextMessageEndEvent::new()));
            self.thinking_state.text_started = false;
        }

        // Close thinking if open
        if self.thinking_state.started {
            events.push(Box::new(ThinkingEndEvent::new()));
            self.thinking_state.started = false;
        }

        // Close text message if open
        if self.text_state.started {
            if let Some(ref msg_id) = self.text_state.message_id {
                events.push(Box::new(TextMessageEndEvent::new(msg_id)));
            }
            self.text_state.started = false;
        }

        // Only emit RunFinished if we haven't had an error
        if !self.had_error {
            events.push(Box::new(RunFinishedEvent::new(
                &self.thread_id,
                &self.run_id,
            )));
        }

        events
    }

    /// Handle an error and generate error events.
    pub fn on_error(&mut self, error: &str) -> Vec<Box<dyn Event>> {
        self.had_error = true;

        let mut events: Vec<Box<dyn Event>> = Vec::new();

        // Close any open streams
        if self.thinking_state.text_started {
            events.push(Box::new(ThinkingTextMessageEndEvent::new()));
            self.thinking_state.text_started = false;
        }
        if self.thinking_state.started {
            events.push(Box::new(ThinkingEndEvent::new()));
            self.thinking_state.started = false;
        }
        if self.text_state.started {
            if let Some(ref msg_id) = self.text_state.message_id {
                events.push(Box::new(TextMessageEndEvent::new(msg_id)));
            }
            self.text_state.started = false;
        }

        events.push(Box::new(RunErrorEvent::new(error)));
        events
    }

    /// Transform an agent stream event into AG-UI events.
    pub fn transform_event<O>(&mut self, event: AgentStreamEvent<O>) -> Vec<Box<dyn Event>> {
        match event {
            AgentStreamEvent::RunStart { .. } => {
                // Handled by before_stream()
                vec![]
            }

            AgentStreamEvent::RequestStart { .. } => {
                // New request might need to close previous messages
                self.handle_request_start()
            }

            AgentStreamEvent::TextDelta { content, part_index } => {
                self.handle_text_delta(content, part_index)
            }

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

            AgentStreamEvent::ResponseComplete { .. }
            | AgentStreamEvent::UsageUpdate { .. }
            | AgentStreamEvent::PartialOutput { .. }
            | AgentStreamEvent::FinalOutput { .. } => {
                // These don't map to AG-UI events
                vec![]
            }

            AgentStreamEvent::RunComplete { .. } => {
                // Handled by after_stream()
                vec![]
            }

            AgentStreamEvent::Error { message, .. } => self.on_error(&message),
        }
    }

    /// Handle request start (may need to close previous message).
    fn handle_request_start(&mut self) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        // Close thinking if open
        if self.thinking_state.text_started {
            events.push(Box::new(ThinkingTextMessageEndEvent::new()));
            self.thinking_state.text_started = false;
        }
        if self.thinking_state.started {
            events.push(Box::new(ThinkingEndEvent::new()));
            self.thinking_state.started = false;
        }

        // Close text message if open
        if self.text_state.started {
            if let Some(ref msg_id) = self.text_state.message_id {
                events.push(Box::new(TextMessageEndEvent::new(msg_id)));
            }
            self.text_state = TextMessageState::default();
        }

        // Reset for new message
        self.current_message_id = None;

        events
    }

    /// Handle text delta event.
    fn handle_text_delta(&mut self, content: String, part_index: usize) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        // Close thinking if we're switching to text
        if self.thinking_state.text_started {
            events.push(Box::new(ThinkingTextMessageEndEvent::new()));
            self.thinking_state.text_started = false;
        }
        if self.thinking_state.started {
            events.push(Box::new(ThinkingEndEvent::new()));
            self.thinking_state.started = false;
        }

        // Check if we need to start a new message or part changed
        let need_new_message = !self.text_state.started
            || self.text_state.part_index != Some(part_index);

        if need_new_message {
            // Close previous message if exists
            if self.text_state.started {
                if let Some(ref msg_id) = self.text_state.message_id {
                    events.push(Box::new(TextMessageEndEvent::new(msg_id)));
                }
            }

            // Start new message
            let message_id = self.new_message_id();
            self.text_state = TextMessageState {
                message_id: Some(message_id.clone()),
                started: true,
                part_index: Some(part_index),
            };
            self.current_message_id = Some(message_id.clone());

            events.push(Box::new(TextMessageStartEvent::new(&message_id)));
        }

        // Emit content
        if let Some(ref msg_id) = self.text_state.message_id {
            events.push(Box::new(TextMessageContentEvent::new(msg_id, content)));
        }

        events
    }

    /// Handle thinking delta event.
    fn handle_thinking_delta(&mut self, content: String, index: usize) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        // Close text message if we're switching to thinking
        if self.text_state.started {
            if let Some(ref msg_id) = self.text_state.message_id {
                events.push(Box::new(TextMessageEndEvent::new(msg_id)));
            }
            self.text_state = TextMessageState::default();
        }

        // Check if thinking index changed
        let need_new_thinking = !self.thinking_state.started
            || self.thinking_state.index != Some(index);

        if need_new_thinking {
            // Close previous thinking if exists
            if self.thinking_state.text_started {
                events.push(Box::new(ThinkingTextMessageEndEvent::new()));
                self.thinking_state.text_started = false;
            }
            if self.thinking_state.started {
                events.push(Box::new(ThinkingEndEvent::new()));
            }

            // Start new thinking
            self.thinking_state = ThinkingState {
                started: true,
                text_started: false,
                index: Some(index),
            };

            events.push(Box::new(ThinkingStartEvent::new()));
        }

        // Start thinking text if not started
        if !self.thinking_state.text_started {
            events.push(Box::new(ThinkingTextMessageStartEvent::new()));
            self.thinking_state.text_started = true;
        }

        // Emit content
        events.push(Box::new(ThinkingTextMessageContentEvent::new(content)));

        events
    }

    /// Handle tool call start.
    fn handle_tool_call_start(
        &mut self,
        name: String,
        tool_call_id: Option<String>,
        index: usize,
    ) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        // Close thinking if open
        if self.thinking_state.text_started {
            events.push(Box::new(ThinkingTextMessageEndEvent::new()));
            self.thinking_state.text_started = false;
        }
        if self.thinking_state.started {
            events.push(Box::new(ThinkingEndEvent::new()));
            self.thinking_state.started = false;
        }

        // Close text message if open
        if self.text_state.started {
            if let Some(ref msg_id) = self.text_state.message_id {
                events.push(Box::new(TextMessageEndEvent::new(msg_id)));
            }
            self.text_state = TextMessageState::default();
        }

        // Generate tool call ID if not provided
        let call_id = tool_call_id.unwrap_or_else(|| format!("call-{}", index));

        // Store state
        let parent_message_id = self.current_message_id.clone();
        self.tool_calls.insert(
            index,
            ToolCallState {
                tool_call_id: call_id.clone(),
                tool_name: name.clone(),
                parent_message_id: parent_message_id.clone(),
                args_started: false,
            },
        );
        self.tool_call_id_to_index.insert(call_id.clone(), index);

        // Create event
        let mut event = ToolCallStartEvent::new(&call_id, &name);
        if let Some(ref parent_id) = parent_message_id {
            event = event.with_parent_message_id(parent_id);
        }

        events.push(Box::new(event));
        events
    }

    /// Handle tool call arguments delta.
    fn handle_tool_call_delta(
        &mut self,
        args_delta: String,
        index: usize,
    ) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        if let Some(state) = self.tool_calls.get_mut(&index) {
            state.args_started = true;
            events.push(Box::new(ToolCallArgsEvent::new(
                &state.tool_call_id,
                args_delta,
            )));
        }

        events
    }

    /// Handle tool call complete.
    fn handle_tool_call_complete(
        &mut self,
        _name: String,
        _args: serde_json::Value,
        index: usize,
    ) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        if let Some(state) = self.tool_calls.get(&index) {
            events.push(Box::new(ToolCallEndEvent::new(&state.tool_call_id)));
        }

        events
    }

    /// Handle tool result.
    fn handle_tool_result(
        &mut self,
        result: serde_json::Value,
        success: bool,
        index: usize,
    ) -> Vec<Box<dyn Event>> {
        let mut events: Vec<Box<dyn Event>> = Vec::new();

        if let Some(state) = self.tool_calls.get(&index) {
            let event = if success {
                ToolCallResultEvent::success(&state.tool_call_id, result)
            } else {
                let error_msg = result
                    .as_str()
                    .unwrap_or("Tool execution failed")
                    .to_string();
                ToolCallResultEvent::error(&state.tool_call_id, error_msg)
            };
            events.push(Box::new(event));
        }

        // Clean up
        if let Some(state) = self.tool_calls.remove(&index) {
            self.tool_call_id_to_index.remove(&state.tool_call_id);
        }

        events
    }
}

/// Helper to create output from events.
///
/// Concatenates all events in the specified format.
pub fn events_to_output(events: &[Box<dyn Event>], format: OutputFormat) -> String {
    events
        .iter()
        .map(|e| AgUiEventStream::encode_event(&**e, format))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_event_stream() {
        let stream = AgUiEventStream::new("thread-1", "run-1");
        assert_eq!(stream.thread_id, "thread-1");
        assert_eq!(stream.run_id, "run-1");
        assert!(!stream.had_error);
    }

    #[test]
    fn test_before_stream() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        let events = stream.before_stream();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type(), EventType::RunStarted);
    }

    #[test]
    fn test_after_stream() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        stream.before_stream();
        let events = stream.after_stream();

        assert!(events.iter().any(|e| e.event_type() == EventType::RunFinished));
    }

    #[test]
    fn test_text_delta_transformation() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        stream.before_stream();

        // First text delta should emit start + content
        let event: AgentStreamEvent<()> = AgentStreamEvent::TextDelta {
            content: "Hello".to_string(),
            part_index: 0,
        };
        let events = stream.transform_event(event);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type(), EventType::TextMessageStart);
        assert_eq!(events[1].event_type(), EventType::TextMessageContent);

        // Second delta should only emit content
        let event2: AgentStreamEvent<()> = AgentStreamEvent::TextDelta {
            content: " World".to_string(),
            part_index: 0,
        };
        let events2 = stream.transform_event(event2);

        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type(), EventType::TextMessageContent);
    }

    #[test]
    fn test_thinking_delta_transformation() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        stream.before_stream();

        let event: AgentStreamEvent<()> = AgentStreamEvent::ThinkingDelta {
            content: "Let me think...".to_string(),
            index: 0,
        };
        let events = stream.transform_event(event);

        // Should emit: ThinkingStart, ThinkingTextMessageStart, ThinkingTextMessageContent
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type(), EventType::ThinkingStart);
        assert_eq!(events[1].event_type(), EventType::ThinkingTextMessageStart);
        assert_eq!(events[2].event_type(), EventType::ThinkingTextMessageContent);
    }

    #[test]
    fn test_tool_call_flow() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        stream.before_stream();

        // Tool call start
        let start_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallStart {
            name: "get_weather".to_string(),
            tool_call_id: Some("call-123".to_string()),
            index: 0,
        };
        let events = stream.transform_event(start_event);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type(), EventType::ToolCallStart);

        // Tool call args
        let args_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallDelta {
            args_delta: "{\"city\":".to_string(),
            index: 0,
        };
        let events2 = stream.transform_event(args_event);

        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type(), EventType::ToolCallArgs);

        // Tool call complete
        let complete_event: AgentStreamEvent<()> = AgentStreamEvent::ToolCallComplete {
            name: "get_weather".to_string(),
            args: serde_json::json!({"city": "London"}),
            index: 0,
        };
        let events3 = stream.transform_event(complete_event);

        assert_eq!(events3.len(), 1);
        assert_eq!(events3[0].event_type(), EventType::ToolCallEnd);

        // Tool result
        let result_event: AgentStreamEvent<()> = AgentStreamEvent::ToolResult {
            name: "get_weather".to_string(),
            result: serde_json::json!({"temp": 20}),
            success: true,
            index: 0,
        };
        let events4 = stream.transform_event(result_event);

        assert_eq!(events4.len(), 1);
        assert_eq!(events4[0].event_type(), EventType::ToolCallResult);
    }

    #[test]
    fn test_on_error() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        stream.before_stream();

        let events = stream.on_error("Something went wrong");

        assert!(stream.had_error);
        assert!(events.iter().any(|e| e.event_type() == EventType::RunError));

        // after_stream should not emit RunFinished
        let after_events = stream.after_stream();
        assert!(!after_events.iter().any(|e| e.event_type() == EventType::RunFinished));
    }

    #[test]
    fn test_encode_event_sse() {
        let event = TextMessageContentEvent::new("msg-1", "Hello");
        let encoded = AgUiEventStream::encode_event(&event, OutputFormat::Sse);

        assert!(encoded.starts_with("data: "));
        assert!(encoded.ends_with("\n\n"));
        assert!(encoded.contains("TEXT_MESSAGE_CONTENT"));
    }

    #[test]
    fn test_encode_event_ndjson() {
        let event = TextMessageContentEvent::new("msg-1", "Hello");
        let encoded = AgUiEventStream::encode_event(&event, OutputFormat::Ndjson);

        assert!(!encoded.starts_with("data: "));
        assert!(encoded.ends_with('\n'));
        assert!(encoded.contains("TEXT_MESSAGE_CONTENT"));
    }

    #[test]
    fn test_events_to_output() {
        let events: Vec<Box<dyn Event>> = vec![
            Box::new(ThinkingStartEvent::new()),
            Box::new(ThinkingTextMessageContentEvent::new("thinking...")),
        ];

        let sse = events_to_output(&events, OutputFormat::Sse);
        assert_eq!(sse.matches("data: ").count(), 2);

        let ndjson = events_to_output(&events, OutputFormat::Ndjson);
        assert!(!ndjson.contains("data: "));
        assert_eq!(ndjson.matches('\n').count(), 2);
    }

    #[test]
    fn test_switching_from_thinking_to_text() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");
        stream.before_stream();

        // Start with thinking
        let thinking: AgentStreamEvent<()> = AgentStreamEvent::ThinkingDelta {
            content: "Let me think".to_string(),
            index: 0,
        };
        stream.transform_event(thinking);

        // Switch to text
        let text: AgentStreamEvent<()> = AgentStreamEvent::TextDelta {
            content: "Here's the answer".to_string(),
            part_index: 0,
        };
        let events = stream.transform_event(text);

        // Should close thinking first
        assert!(events.iter().any(|e| e.event_type() == EventType::ThinkingTextMessageEnd));
        assert!(events.iter().any(|e| e.event_type() == EventType::ThinkingEnd));
        assert!(events.iter().any(|e| e.event_type() == EventType::TextMessageStart));
    }

    #[test]
    fn test_full_lifecycle() {
        let mut stream = AgUiEventStream::new("thread-1", "run-1");

        // Before
        let before = stream.before_stream();
        assert!(!before.is_empty());

        // Some text
        let text: AgentStreamEvent<()> = AgentStreamEvent::text_delta("Hello", 0);
        let text_events = stream.transform_event(text);
        assert!(!text_events.is_empty());

        // After
        let after = stream.after_stream();
        assert!(after.iter().any(|e| e.event_type() == EventType::TextMessageEnd));
        assert!(after.iter().any(|e| e.event_type() == EventType::RunFinished));
    }
}
