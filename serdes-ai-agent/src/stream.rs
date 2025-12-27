//! Streaming agent execution.
//!
//! This module provides streaming support for agent runs.

use crate::agent::Agent;
use crate::context::{generate_run_id, RunContext, RunUsage};
use crate::errors::AgentRunError;
use crate::run::{AgentRunResult, RunOptions};
use chrono::Utc;
use futures::Stream;
use pin_project_lite::pin_project;
use serdes_ai_core::messages::{ModelResponseStreamEvent, UserContent};
use serdes_ai_core::{ModelRequest, ModelResponse, ModelSettings};
use serdes_ai_models::ModelRequestParameters;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Events emitted during streaming.
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    /// Run started.
    RunStart {
        run_id: String,
    },
    /// Model request started.
    RequestStart {
        step: u32,
    },
    /// Text delta.
    TextDelta {
        text: String,
    },
    /// Tool call started.
    ToolCallStart {
        tool_name: String,
        tool_call_id: Option<String>,
    },
    /// Tool call arguments delta.
    ToolCallDelta {
        delta: String,
    },
    /// Tool call completed.
    ToolCallComplete {
        tool_name: String,
    },
    /// Tool executed.
    ToolExecuted {
        tool_name: String,
        success: bool,
    },
    /// Thinking delta (for reasoning models).
    ThinkingDelta {
        text: String,
    },
    /// Model response completed.
    ResponseComplete {
        step: u32,
    },
    /// Output ready.
    OutputReady,
    /// Run completed.
    RunComplete {
        run_id: String,
    },
    /// Error occurred.
    Error {
        message: String,
    },
}

/// Streaming agent execution.
///
/// This is a placeholder implementation. Full streaming requires more complex
/// state management to handle tool calls and retries during streaming.
pub struct AgentStream<'a, Deps, Output> {
    #[allow(dead_code)]
    agent: &'a Agent<Deps, Output>,
    #[allow(dead_code)]
    deps: Arc<Deps>,
    #[allow(dead_code)]
    ctx: RunContext<Deps>,
    run_id: String,
    started: bool,
    finished: bool,
}

impl<'a, Deps, Output> AgentStream<'a, Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create a new streaming agent run.
    pub async fn new(
        agent: &'a Agent<Deps, Output>,
        prompt: UserContent,
        deps: Deps,
        options: RunOptions,
    ) -> Result<Self, AgentRunError> {
        let run_id = generate_run_id();
        let deps = Arc::new(deps);

        let model_settings = options
            .model_settings
            .unwrap_or_else(|| agent.model_settings.clone());

        let ctx = RunContext {
            deps: deps.clone(),
            run_id: run_id.clone(),
            start_time: Utc::now(),
            model_name: agent.model().name().to_string(),
            model_settings: model_settings.clone(),
            tool_name: None,
            tool_call_id: None,
            retry_count: 0,
            metadata: options.metadata.clone(),
        };

        // For now, streaming is not fully implemented
        // A full implementation would use the model's streaming API
        let _ = prompt; // Suppress unused warning

        Ok(Self {
            agent,
            deps,
            ctx,
            run_id,
            started: false,
            finished: false,
        })
    }
}

impl<'a, Deps, Output> Stream for AgentStream<'a, Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    type Item = Result<AgentStreamEvent, AgentRunError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        if !self.started {
            self.started = true;
            return Poll::Ready(Some(Ok(AgentStreamEvent::RunStart {
                run_id: self.run_id.clone(),
            })));
        }

        // Mark as finished - full implementation would process model stream here
        self.finished = true;
        Poll::Ready(Some(Ok(AgentStreamEvent::RunComplete {
            run_id: self.run_id.clone(),
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_debug() {
        let event = AgentStreamEvent::TextDelta {
            text: "hello".to_string(),
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("TextDelta"));
    }

    #[test]
    fn test_stream_event_variants() {
        let events = vec![
            AgentStreamEvent::RunStart {
                run_id: "123".to_string(),
            },
            AgentStreamEvent::RequestStart { step: 1 },
            AgentStreamEvent::TextDelta {
                text: "hi".to_string(),
            },
            AgentStreamEvent::ToolCallStart {
                tool_name: "search".to_string(),
                tool_call_id: Some("call-1".to_string()),
            },
            AgentStreamEvent::OutputReady,
            AgentStreamEvent::RunComplete {
                run_id: "123".to_string(),
            },
        ];

        assert_eq!(events.len(), 6);
    }
}
