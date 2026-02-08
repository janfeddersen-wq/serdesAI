//! Claude Code OAuth types.
//!
//! These are API request/response types for the Claude Code OAuth flow.

#![allow(missing_docs)] // DTO fields are self-documenting

use serde::{Deserialize, Serialize};

/// Claude Code API configuration.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Base URL for the API
    pub api_base_url: String,
    /// Model prefix for display
    pub prefix: String,
    /// Default context length
    pub context_length: usize,
    /// Anthropic API version header
    pub anthropic_version: String,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            api_base_url: "https://api.anthropic.com".to_string(),
            prefix: "claude-code-".to_string(),
            context_length: 200000,
            anthropic_version: "2023-06-01".to_string(),
        }
    }
}

/// Extended thinking configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ThinkingConfig {
    /// Thinking type: "enabled".
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// Budget tokens for thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
}

impl ThinkingConfig {
    /// Create enabled thinking config without a budget.
    pub fn enabled() -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens: None,
        }
    }

    /// Create thinking config with a specific budget.
    pub fn with_budget(budget: u64) -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(budget),
        }
    }
}

/// Request body for Claude messages API.
#[derive(Debug, Serialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Extended thinking configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// Message in a Claude request.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: ClaudeContent,
}

/// Message content (string or content blocks).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ClaudeContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Cache control for prompt caching.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

/// Content block for multi-modal messages.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "document")]
    Document { source: DocumentSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Source for document content (e.g. PDF).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Tool definition for Claude.
#[derive(Debug, Serialize, Clone)]
pub struct ClaudeTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Response from Claude messages API.
#[derive(Debug, Deserialize)]
pub struct ClaudeResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<ClaudeUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

#[derive(Debug, Deserialize, Default)]
#[allow(missing_docs)]
pub struct ClaudeUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}
