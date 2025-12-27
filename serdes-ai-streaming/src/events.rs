//! Streaming event types.
//!
//! This module defines the events emitted during agent streaming.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serdes_ai_core::{ModelResponse, RequestUsage};
use std::fmt;

/// Events emitted during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentStreamEvent<Output = JsonValue> {
    /// Run started.
    RunStart {
        /// The run ID.
        run_id: String,
        /// Step number (0 for start).
        step: u32,
    },

    /// Model request started.
    RequestStart {
        /// Current step number.
        step: u32,
    },

    /// Text delta received.
    TextDelta {
        /// The text content.
        content: String,
        /// Index of the part this delta belongs to.
        part_index: usize,
    },

    /// Tool call started.
    ToolCallStart {
        /// Tool name.
        name: String,
        /// Tool call ID (if available).
        tool_call_id: Option<String>,
        /// Index of this tool call.
        index: usize,
    },

    /// Tool call arguments delta.
    ToolCallDelta {
        /// Arguments delta (JSON string).
        args_delta: String,
        /// Index of this tool call.
        index: usize,
    },

    /// Tool call completed.
    ToolCallComplete {
        /// Tool name.
        name: String,
        /// Complete arguments.
        args: JsonValue,
        /// Index of this tool call.
        index: usize,
    },

    /// Tool result received.
    ToolResult {
        /// Tool name.
        name: String,
        /// Result content.
        result: JsonValue,
        /// Whether the tool succeeded.
        success: bool,
        /// Index of this tool call.
        index: usize,
    },

    /// Thinking delta (for Claude extended thinking).
    ThinkingDelta {
        /// The thinking content.
        content: String,
        /// Index of this thinking part.
        index: usize,
    },

    /// Partial output available (validated incrementally).
    PartialOutput {
        /// The partial output.
        output: Output,
    },

    /// Model response complete.
    ResponseComplete {
        /// The complete response.
        response: ModelResponse,
    },

    /// Usage update.
    UsageUpdate {
        /// Current usage.
        usage: RequestUsage,
    },

    /// Final output ready.
    FinalOutput {
        /// The final output.
        output: Output,
    },

    /// Run complete.
    RunComplete {
        /// The run ID.
        run_id: String,
        /// Total steps.
        total_steps: u32,
    },

    /// Error occurred.
    Error {
        /// Error message.
        message: String,
        /// Whether the error is recoverable.
        recoverable: bool,
    },
}

impl<Output> AgentStreamEvent<Output> {
    /// Create a run start event.
    pub fn run_start(run_id: impl Into<String>, step: u32) -> Self {
        Self::RunStart {
            run_id: run_id.into(),
            step,
        }
    }

    /// Create a text delta event.
    pub fn text_delta(content: impl Into<String>, part_index: usize) -> Self {
        Self::TextDelta {
            content: content.into(),
            part_index,
        }
    }

    /// Create an error event.
    pub fn error(message: impl Into<String>, recoverable: bool) -> Self {
        Self::Error {
            message: message.into(),
            recoverable,
        }
    }

    /// Check if this is the final event.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::RunComplete { .. } | Self::Error { .. })
    }

    /// Check if this is an error event.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the text content if this is a text delta.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextDelta { content, .. } => Some(content),
            _ => None,
        }
    }

    /// Get the output if this is a final output event.
    pub fn as_output(&self) -> Option<&Output> {
        match self {
            Self::FinalOutput { output } => Some(output),
            Self::PartialOutput { output } => Some(output),
            _ => None,
        }
    }

    /// Map the output type.
    pub fn map_output<U, F>(self, f: F) -> AgentStreamEvent<U>
    where
        F: FnOnce(Output) -> U,
    {
        match self {
            Self::RunStart { run_id, step } => AgentStreamEvent::RunStart { run_id, step },
            Self::RequestStart { step } => AgentStreamEvent::RequestStart { step },
            Self::TextDelta { content, part_index } => {
                AgentStreamEvent::TextDelta { content, part_index }
            }
            Self::ToolCallStart {
                name,
                tool_call_id,
                index,
            } => AgentStreamEvent::ToolCallStart {
                name,
                tool_call_id,
                index,
            },
            Self::ToolCallDelta { args_delta, index } => {
                AgentStreamEvent::ToolCallDelta { args_delta, index }
            }
            Self::ToolCallComplete { name, args, index } => {
                AgentStreamEvent::ToolCallComplete { name, args, index }
            }
            Self::ToolResult {
                name,
                result,
                success,
                index,
            } => AgentStreamEvent::ToolResult {
                name,
                result,
                success,
                index,
            },
            Self::ThinkingDelta { content, index } => {
                AgentStreamEvent::ThinkingDelta { content, index }
            }
            Self::PartialOutput { output } => AgentStreamEvent::PartialOutput { output: f(output) },
            Self::ResponseComplete { response } => AgentStreamEvent::ResponseComplete { response },
            Self::UsageUpdate { usage } => AgentStreamEvent::UsageUpdate { usage },
            Self::FinalOutput { output } => AgentStreamEvent::FinalOutput { output: f(output) },
            Self::RunComplete {
                run_id,
                total_steps,
            } => AgentStreamEvent::RunComplete {
                run_id,
                total_steps,
            },
            Self::Error {
                message,
                recoverable,
            } => AgentStreamEvent::Error {
                message,
                recoverable,
            },
        }
    }
}

impl<Output: fmt::Display> fmt::Display for AgentStreamEvent<Output> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RunStart { run_id, .. } => write!(f, "[run_start] {}", run_id),
            Self::RequestStart { step } => write!(f, "[request_start] step {}", step),
            Self::TextDelta { content, .. } => write!(f, "{}", content),
            Self::ToolCallStart { name, .. } => write!(f, "[tool_start] {}", name),
            Self::ToolCallDelta { args_delta, .. } => write!(f, "{}", args_delta),
            Self::ToolCallComplete { name, .. } => write!(f, "[tool_complete] {}", name),
            Self::ToolResult { name, success, .. } => {
                write!(f, "[tool_result] {} ({})", name, if *success { "ok" } else { "error" })
            }
            Self::ThinkingDelta { content, .. } => write!(f, "[thinking] {}", content),
            Self::PartialOutput { output } => write!(f, "[partial] {}", output),
            Self::ResponseComplete { .. } => write!(f, "[response_complete]"),
            Self::UsageUpdate { .. } => write!(f, "[usage_update]"),
            Self::FinalOutput { output } => write!(f, "[output] {}", output),
            Self::RunComplete { run_id, .. } => write!(f, "[run_complete] {}", run_id),
            Self::Error { message, .. } => write!(f, "[error] {}", message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event: AgentStreamEvent<String> = AgentStreamEvent::run_start("run-123", 0);
        assert!(!event.is_terminal());
        assert!(!event.is_error());
    }

    #[test]
    fn test_text_delta() {
        let event: AgentStreamEvent<String> = AgentStreamEvent::text_delta("Hello", 0);
        assert_eq!(event.as_text(), Some("Hello"));
    }

    #[test]
    fn test_terminal_events() {
        let complete: AgentStreamEvent<String> = AgentStreamEvent::RunComplete {
            run_id: "run-123".to_string(),
            total_steps: 1,
        };
        assert!(complete.is_terminal());

        let error: AgentStreamEvent<String> = AgentStreamEvent::error("oops", false);
        assert!(error.is_terminal());
        assert!(error.is_error());
    }

    #[test]
    fn test_map_output() {
        let event: AgentStreamEvent<i32> = AgentStreamEvent::FinalOutput { output: 42 };
        let mapped = event.map_output(|n| n.to_string());
        
        if let AgentStreamEvent::FinalOutput { output } = mapped {
            assert_eq!(output, "42");
        } else {
            panic!("Expected FinalOutput");
        }
    }

    #[test]
    fn test_display() {
        let event: AgentStreamEvent<String> = AgentStreamEvent::text_delta("test", 0);
        assert_eq!(format!("{}", event), "test");
    }
}
