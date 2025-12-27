//! Agent run results.

use serde::de::DeserializeOwned;
use serdes_ai_core::{ModelResponse, Usage, Error, Result};
use crate::history::ConversationHistory;

/// Result of an agent run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// The final model response.
    pub response: ModelResponse,
    /// Conversation history.
    pub history: ConversationHistory,
    /// Total token usage across all turns.
    pub total_usage: Usage,
}

impl RunResult {
    /// Create a new run result.
    pub fn new(response: ModelResponse, history: ConversationHistory) -> Self {
        let total_usage = response.usage.clone();
        Self {
            response,
            history,
            total_usage,
        }
    }

    /// Get the output text.
    pub fn output(&self) -> &str {
        &self.response.content
    }

    /// Parse the output as a typed value.
    pub fn into_typed<T: DeserializeOwned>(self) -> Result<T> {
        serde_json::from_str(&self.response.content)
            .map_err(|e| Error::ValidationError(e.to_string()))
    }

    /// Get the number of turns in the conversation.
    pub fn turn_count(&self) -> usize {
        self.history.len()
    }
}

/// Result of a streamed agent run.
pub struct StreamedRunResult {
    /// Final result after stream completion.
    pub result: RunResult,
}
