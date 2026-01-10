//! Agent streaming implementation.
//!
//! This module provides the `AgentStream` type for streaming agent execution.

use crate::error::StreamResult;
use crate::events::AgentStreamEvent;
use crate::partial_response::{PartialResponse, ResponseDelta};
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;
use serde::de::DeserializeOwned;
use serdes_ai_core::{ModelResponse, RequestUsage};
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

/// State of the agent stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Not started yet.
    Pending,
    /// Currently streaming from model.
    Streaming,
    /// Processing tool calls.
    ProcessingTools,
    /// Waiting for retry.
    Retrying,
    /// Successfully completed.
    Completed,
    /// Failed with error.
    Failed,
}

/// Configuration for agent streaming.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Whether to emit partial outputs.
    pub emit_partial_outputs: bool,
    /// Minimum interval between partial output emissions (ms).
    pub partial_output_interval_ms: u64,
    /// Whether to emit thinking deltas.
    pub emit_thinking: bool,
    /// Whether to accumulate tool arguments before emitting.
    pub buffer_tool_args: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            emit_partial_outputs: true,
            partial_output_interval_ms: 100,
            emit_thinking: true,
            buffer_tool_args: false,
        }
    }
}

pin_project! {
    /// Streaming agent execution.
    ///
    /// This struct wraps the streaming execution of an agent run,
    /// emitting events as the model generates responses.
    pub struct AgentStream<S, Output> {
        #[pin]
        inner: S,
        run_id: String,
        step: u32,
        state: StreamState,
        config: StreamConfig,
        partial_response: PartialResponse,
        pending_events: VecDeque<AgentStreamEvent<Output>>,
        accumulated_usage: RequestUsage,
        _output: std::marker::PhantomData<Output>,
    }
}

impl<S, Output> AgentStream<S, Output>
where
    S: Stream<Item = StreamResult<ResponseDelta>>,
    Output: DeserializeOwned,
{
    /// Create a new agent stream.
    pub fn new(inner: S, run_id: impl Into<String>) -> Self {
        let run_id = run_id.into();
        Self {
            inner,
            run_id: run_id.clone(),
            step: 0,
            state: StreamState::Pending,
            config: StreamConfig::default(),
            partial_response: PartialResponse::new(),
            pending_events: VecDeque::new(),
            accumulated_usage: RequestUsage::new(),
            _output: std::marker::PhantomData,
        }
    }

    /// Set the stream configuration.
    pub fn with_config(mut self, config: StreamConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the run ID.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get the current step.
    pub fn step(&self) -> u32 {
        self.step
    }

    /// Get the current state.
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Get the current partial response.
    pub fn partial_response(&self) -> &PartialResponse {
        &self.partial_response
    }

    /// Get the accumulated response as a snapshot.
    pub fn response_snapshot(&self) -> ModelResponse {
        self.partial_response.as_response()
    }

    /// Get the accumulated text content.
    pub fn text_content(&self) -> String {
        self.partial_response.text_content()
    }

    /// Get accumulated usage.
    pub fn usage(&self) -> &RequestUsage {
        &self.accumulated_usage
    }

    /// Check if the stream is complete.
    pub fn is_complete(&self) -> bool {
        matches!(self.state, StreamState::Completed | StreamState::Failed)
    }

    #[allow(dead_code)]
    fn process_delta(&mut self, delta: ResponseDelta) {
        match &delta {
            ResponseDelta::Text { index, content } => {
                self.pending_events.push_back(AgentStreamEvent::TextDelta {
                    content: content.clone(),
                    part_index: *index,
                });
            }
            ResponseDelta::ToolCall {
                index,
                name,
                args,
                id,
            } => {
                // Emit tool call start if we have a name
                if let Some(name) = name {
                    self.pending_events
                        .push_back(AgentStreamEvent::ToolCallStart {
                            name: name.clone(),
                            tool_call_id: id.clone(),
                            index: *index,
                        });
                }

                // Emit args delta if we have args
                if let Some(args) = args {
                    if !self.config.buffer_tool_args {
                        self.pending_events
                            .push_back(AgentStreamEvent::ToolCallDelta {
                                args_delta: args.clone(),
                                index: *index,
                            });
                    }
                }
            }
            ResponseDelta::Thinking { index, content, .. } => {
                if self.config.emit_thinking {
                    self.pending_events
                        .push_back(AgentStreamEvent::ThinkingDelta {
                            content: content.clone(),
                            index: *index,
                        });
                }
            }
            ResponseDelta::Finish { .. } => {
                self.state = StreamState::Completed;
            }
            ResponseDelta::Usage { usage } => {
                self.accumulated_usage = self.accumulated_usage.clone() + usage.clone();
                self.pending_events
                    .push_back(AgentStreamEvent::UsageUpdate {
                        usage: self.accumulated_usage.clone(),
                    });
            }
        }

        // Apply delta to partial response
        self.partial_response.apply_delta(&delta);
    }
}

impl<S, Output> Stream for AgentStream<S, Output>
where
    S: Stream<Item = StreamResult<ResponseDelta>> + Unpin,
    Output: DeserializeOwned + Clone,
{
    type Item = AgentStreamEvent<Output>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Return pending events first
        if let Some(event) = this.pending_events.pop_front() {
            return Poll::Ready(Some(event));
        }

        // Check if completed
        if matches!(this.state, StreamState::Completed | StreamState::Failed) {
            return Poll::Ready(None);
        }

        // Emit run start if pending
        if *this.state == StreamState::Pending {
            *this.state = StreamState::Streaming;
            *this.step += 1;
            return Poll::Ready(Some(AgentStreamEvent::RunStart {
                run_id: this.run_id.clone(),
                step: *this.step,
            }));
        }

        // Poll the inner stream
        match this.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(delta))) => {
                // Process the delta
                match &delta {
                    ResponseDelta::Text { index, content } => {
                        this.pending_events.push_back(AgentStreamEvent::TextDelta {
                            content: content.clone(),
                            part_index: *index,
                        });
                    }
                    ResponseDelta::ToolCall {
                        index,
                        name,
                        args,
                        id,
                    } => {
                        if let Some(name) = name {
                            this.pending_events
                                .push_back(AgentStreamEvent::ToolCallStart {
                                    name: name.clone(),
                                    tool_call_id: id.clone(),
                                    index: *index,
                                });
                        }
                        if let Some(args) = args {
                            if !this.config.buffer_tool_args {
                                this.pending_events
                                    .push_back(AgentStreamEvent::ToolCallDelta {
                                        args_delta: args.clone(),
                                        index: *index,
                                    });
                            }
                        }
                    }
                    ResponseDelta::Thinking { index, content, .. } => {
                        if this.config.emit_thinking {
                            this.pending_events
                                .push_back(AgentStreamEvent::ThinkingDelta {
                                    content: content.clone(),
                                    index: *index,
                                });
                        }
                    }
                    ResponseDelta::Finish { .. } => {
                        *this.state = StreamState::Completed;
                        this.pending_events
                            .push_back(AgentStreamEvent::ResponseComplete {
                                response: this.partial_response.as_response(),
                            });
                        this.pending_events
                            .push_back(AgentStreamEvent::RunComplete {
                                run_id: this.run_id.clone(),
                                total_steps: *this.step,
                            });
                    }
                    ResponseDelta::Usage { usage } => {
                        *this.accumulated_usage = this.accumulated_usage.clone() + usage.clone();
                        this.pending_events
                            .push_back(AgentStreamEvent::UsageUpdate {
                                usage: this.accumulated_usage.clone(),
                            });
                    }
                }

                // Apply to partial response
                this.partial_response.apply_delta(&delta);

                // Return first pending event
                if let Some(event) = this.pending_events.pop_front() {
                    Poll::Ready(Some(event))
                } else {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(Some(Err(e))) => {
                *this.state = StreamState::Failed;
                Poll::Ready(Some(AgentStreamEvent::Error {
                    message: e.to_string(),
                    recoverable: e.is_recoverable(),
                }))
            }
            Poll::Ready(None) => {
                // Stream ended - finalize if not already done
                if *this.state == StreamState::Streaming {
                    *this.state = StreamState::Completed;
                    this.pending_events
                        .push_back(AgentStreamEvent::ResponseComplete {
                            response: this.partial_response.as_response(),
                        });
                    this.pending_events
                        .push_back(AgentStreamEvent::RunComplete {
                            run_id: this.run_id.clone(),
                            total_steps: *this.step,
                        });

                    if let Some(event) = this.pending_events.pop_front() {
                        return Poll::Ready(Some(event));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Extension trait for creating filtered streams.
pub trait AgentStreamExt<Output>: Stream<Item = AgentStreamEvent<Output>> + Sized {
    /// Filter to only text delta events.
    fn text_deltas(self) -> TextDeltaStream<Self> {
        TextDeltaStream {
            inner: self,
            accumulated: String::new(),
            emit_accumulated: false,
        }
    }

    /// Filter to only text content, accumulating it.
    fn text_accumulated(self) -> TextDeltaStream<Self> {
        TextDeltaStream {
            inner: self,
            accumulated: String::new(),
            emit_accumulated: true,
        }
    }

    /// Filter to only output events.
    fn outputs(self) -> OutputStream<Self, Output> {
        OutputStream {
            inner: self,
            _output: std::marker::PhantomData,
        }
    }

    /// Filter to only response complete events.
    fn responses(self) -> ResponseStream<Self> {
        ResponseStream { inner: self }
    }
}

impl<S, Output> AgentStreamExt<Output> for S where S: Stream<Item = AgentStreamEvent<Output>> {}

/// A text delta with position information.
///
/// This struct is emitted by `TextDeltaStream` when using `text_accumulated()`.
/// It provides incremental text content along with position metadata,
/// avoiding O(n²) string cloning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextDelta {
    /// The actual delta content (just the new text, not the full accumulated string).
    pub content: String,
    /// Position where this delta starts in the accumulated text.
    pub position: usize,
    /// Total length of accumulated text after this delta.
    pub total_length: usize,
}

impl TextDelta {
    /// Create a new text delta.
    pub fn new(content: String, position: usize, total_length: usize) -> Self {
        Self {
            content,
            position,
            total_length,
        }
    }
}

pin_project! {
    /// Stream that filters to text deltas.
    ///
    /// When created via `text_deltas()`, emits just the delta content as `String`.
    /// When created via `text_accumulated()`, emits `TextDelta` with position info.
    pub struct TextDeltaStream<S> {
        #[pin]
        inner: S,
        accumulated: String,
        emit_accumulated: bool,
    }
}

impl<S> TextDeltaStream<S> {
    /// Get the current accumulated text.
    ///
    /// This is useful when you need the full text at the end of streaming.
    pub fn accumulated_text(&self) -> &str {
        &self.accumulated
    }

    /// Consume and return the accumulated text.
    pub fn into_accumulated(self) -> String {
        self.accumulated
    }
}

impl<S, Output> Stream for TextDeltaStream<S>
where
    S: Stream<Item = AgentStreamEvent<Output>>,
{
    // When emit_accumulated is false, we just return the delta content.
    // When true, we still only return the delta content (not the full accumulated),
    // but we track position internally. The caller can use accumulated_text() if needed.
    type Item = TextDelta;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(event)) => match event {
                    AgentStreamEvent::TextDelta { content, .. } => {
                        let position = this.accumulated.len();
                        this.accumulated.push_str(&content);
                        let total_length = this.accumulated.len();

                        // Always emit just the delta, never clone the full accumulated string
                        return Poll::Ready(Some(TextDelta::new(content, position, total_length)));
                    }
                    AgentStreamEvent::RunComplete { .. } | AgentStreamEvent::Error { .. } => {
                        return Poll::Ready(None);
                    }
                    _ => continue, // Skip non-text events
                },
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pin_project! {
    /// Stream that filters to outputs.
    pub struct OutputStream<S, Output> {
        #[pin]
        inner: S,
        _output: std::marker::PhantomData<Output>,
    }
}

impl<S, Output> Stream for OutputStream<S, Output>
where
    S: Stream<Item = AgentStreamEvent<Output>>,
{
    type Item = Output;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(event)) => match event {
                    AgentStreamEvent::FinalOutput { output } => {
                        return Poll::Ready(Some(output));
                    }
                    AgentStreamEvent::PartialOutput { output } => {
                        return Poll::Ready(Some(output));
                    }
                    AgentStreamEvent::RunComplete { .. } | AgentStreamEvent::Error { .. } => {
                        return Poll::Ready(None);
                    }
                    _ => continue,
                },
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pin_project! {
    /// Stream that filters to complete responses.
    pub struct ResponseStream<S> {
        #[pin]
        inner: S,
    }
}

impl<S, Output> Stream for ResponseStream<S>
where
    S: Stream<Item = AgentStreamEvent<Output>>,
{
    type Item = ModelResponse;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(event)) => match event {
                    AgentStreamEvent::ResponseComplete { response } => {
                        return Poll::Ready(Some(response));
                    }
                    AgentStreamEvent::RunComplete { .. } | AgentStreamEvent::Error { .. } => {
                        return Poll::Ready(None);
                    }
                    _ => continue,
                },
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_agent_stream_basic() {
        let deltas = vec![
            Ok(ResponseDelta::Text {
                index: 0,
                content: "Hello".to_string(),
            }),
            Ok(ResponseDelta::Text {
                index: 0,
                content: ", world!".to_string(),
            }),
            Ok(ResponseDelta::Finish {
                reason: serdes_ai_core::FinishReason::Stop,
            }),
        ];

        let inner = stream::iter(deltas);
        let mut agent_stream: AgentStream<_, String> = AgentStream::new(inner, "test-run");

        let mut events = Vec::new();
        while let Some(event) = agent_stream.next().await {
            events.push(event);
        }

        // Should have: RunStart, TextDelta, TextDelta, ResponseComplete, RunComplete
        assert!(events.len() >= 4);
        assert!(matches!(events[0], AgentStreamEvent::RunStart { .. }));
    }

    #[tokio::test]
    async fn test_text_deltas() {
        let deltas = vec![
            Ok(ResponseDelta::Text {
                index: 0,
                content: "Hello".to_string(),
            }),
            Ok(ResponseDelta::Text {
                index: 0,
                content: " world".to_string(),
            }),
            Ok(ResponseDelta::Finish {
                reason: serdes_ai_core::FinishReason::Stop,
            }),
        ];

        let inner = stream::iter(deltas);
        let agent_stream: AgentStream<_, String> = AgentStream::new(inner, "test-run");

        let text_deltas: Vec<TextDelta> = agent_stream.text_deltas().collect().await;

        // Should get individual deltas with position info
        assert_eq!(text_deltas.len(), 2);
        assert_eq!(text_deltas[0].content, "Hello");
        assert_eq!(text_deltas[0].position, 0);
        assert_eq!(text_deltas[0].total_length, 5);
        assert_eq!(text_deltas[1].content, " world");
        assert_eq!(text_deltas[1].position, 5);
        assert_eq!(text_deltas[1].total_length, 11);
    }

    #[tokio::test]
    async fn test_text_accumulated() {
        let deltas = vec![
            Ok(ResponseDelta::Text {
                index: 0,
                content: "Hello".to_string(),
            }),
            Ok(ResponseDelta::Text {
                index: 0,
                content: " world".to_string(),
            }),
            Ok(ResponseDelta::Finish {
                reason: serdes_ai_core::FinishReason::Stop,
            }),
        ];

        let inner = stream::iter(deltas);
        let agent_stream: AgentStream<_, String> = AgentStream::new(inner, "test-run");
        let mut stream = agent_stream.text_accumulated();

        // Collect all deltas
        let text_deltas: Vec<TextDelta> = (&mut stream).collect().await;

        // Each delta only contains the new content, not the full accumulated string
        // This is the O(n²) fix - we no longer clone the full string each time!
        assert_eq!(text_deltas.len(), 2);
        assert_eq!(text_deltas[0].content, "Hello");
        assert_eq!(text_deltas[1].content, " world");

        // The accumulated text can be retrieved via accumulated_text() method
        assert_eq!(stream.accumulated_text(), "Hello world");
    }

    #[tokio::test]
    async fn test_stream_state() {
        let deltas = vec![Ok(ResponseDelta::Text {
            index: 0,
            content: "Test".to_string(),
        })];

        let inner = stream::iter(deltas);
        let agent_stream: AgentStream<_, String> = AgentStream::new(inner, "test-run");

        assert_eq!(agent_stream.state(), StreamState::Pending);
        assert!(!agent_stream.is_complete());
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert!(config.emit_partial_outputs);
        assert!(config.emit_thinking);
        assert!(!config.buffer_tool_args);
    }
}
