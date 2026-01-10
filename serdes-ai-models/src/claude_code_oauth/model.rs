//! Claude Code OAuth model implementation.

use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::{anthropic_claude_profile, ModelProfile};
use async_trait::async_trait;
use base64::Engine;
use reqwest::Client;
use serdes_ai_core::messages::{
    ImageContent, TextPart, ThinkingPart, ToolCallArgs, ToolCallPart, UserContent, UserContentPart,
    UserPromptPart,
};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage,
};
use std::time::Duration;
use tracing::{debug, error, info};

/// The hardcoded system prompt for Claude Code OAuth models.
/// This matches the Python code_puppy implementation.
const CLAUDE_CODE_INSTRUCTIONS: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Tool name prefix for Claude Code OAuth compatibility.
/// Tools must be prefixed on outgoing requests and unprefixed on incoming responses.
const TOOL_PREFIX: &str = "cp_";

/// Claude Code OAuth model.
///
/// Uses OAuth access tokens to authenticate with the Anthropic API.
#[derive(Debug, Clone)]
pub struct ClaudeCodeOAuthModel {
    model_name: String,
    access_token: String,
    client: Client,
    config: ClaudeCodeConfig,
    profile: ModelProfile,
    /// Enable extended thinking.
    enable_thinking: bool,
    /// Thinking budget tokens.
    thinking_budget: Option<u64>,
}

impl ClaudeCodeOAuthModel {
    /// Create a new Claude Code OAuth model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model name (e.g., "claude-sonnet-4-20250514")
    /// * `access_token` - OAuth access token from authentication flow
    pub fn new(model_name: impl Into<String>, access_token: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let access_token = access_token.into();
        let profile = anthropic_claude_profile();
        let config = ClaudeCodeConfig::default();
        let client = Self::build_client();

        Self {
            model_name,
            access_token,
            client,
            config,
            profile,
            enable_thinking: false,
            thinking_budget: None,
        }
    }

    /// Enable extended thinking.
    #[must_use]
    pub fn with_thinking(mut self, budget: Option<u64>) -> Self {
        self.enable_thinking = true;
        self.thinking_budget = budget;
        // Update profile to indicate reasoning support
        self.profile.supports_reasoning = true;
        self
    }

    /// Set a custom config.
    #[must_use]
    pub fn with_config(mut self, config: ClaudeCodeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set a custom HTTP client.
    #[must_use]
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    /// Set a custom profile.
    #[must_use]
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    fn build_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .expect("Failed to build HTTP client")
    }

    /// Convert our messages to Claude format.
    ///
    /// For Claude Code OAuth models, we use special system prompt handling:
    /// 1. The actual system prompt is always the hardcoded CLAUDE_CODE_INSTRUCTIONS
    /// 2. Any user-provided system prompts are prepended to the first user message
    fn convert_messages(&self, requests: &[ModelRequest]) -> (Option<String>, Vec<ClaudeMessage>) {
        // Collect all system prompts to prepend to first user message
        let mut collected_system_prompts: Vec<String> = Vec::new();
        let mut messages = Vec::new();
        let mut first_user_message_processed = false;

        for req in requests {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        // For Claude Code: collect system prompts to prepend to first user message
                        // instead of using them as the actual system prompt
                        collected_system_prompts.push(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        let mut content = self.convert_user_content(user);

                        // Prepend collected system prompts to the FIRST user message only
                        if !first_user_message_processed && !collected_system_prompts.is_empty() {
                            let system_prefix = collected_system_prompts.join("\n\n");
                            content = self.prepend_to_content(content, &system_prefix);
                            first_user_message_processed = true;
                        }

                        messages.push(ClaudeMessage {
                            role: "user".to_string(),
                            content,
                        });
                    }
                    ModelRequestPart::ToolReturn(ret) => {
                        // Tool results go in a user message with tool_result blocks
                        messages.push(ClaudeMessage {
                            role: "user".to_string(),
                            content: ClaudeContent::Blocks(vec![ContentBlock::ToolResult {
                                tool_use_id: ret.tool_call_id.clone().unwrap_or_default(),
                                content: ret.content.to_string_content(),
                                is_error: None,
                            }]),
                        });
                    }
                    ModelRequestPart::RetryPrompt(retry) => {
                        messages.push(ClaudeMessage {
                            role: "user".to_string(),
                            content: ClaudeContent::Text(retry.content.message().to_string()),
                        });
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        messages.push(ClaudeMessage {
                            role: "user".to_string(),
                            content: ClaudeContent::Blocks(vec![ContentBlock::ToolResult {
                                tool_use_id: builtin.tool_call_id.clone(),
                                content: content_str,
                                is_error: None,
                            }]),
                        });
                    }
                    ModelRequestPart::ModelResponse(response) => {
                        // Add the assistant response to messages for proper alternation
                        self.add_response_to_messages(&mut messages, response);
                    }
                }
            }
        }

        // Always use the hardcoded Claude Code instructions as the system prompt
        let system_prompt = Some(CLAUDE_CODE_INSTRUCTIONS.to_string());

        (system_prompt, messages)
    }

    /// Helper to prepend text to ClaudeContent.
    /// Used to prepend user-provided system prompts to the first user message.
    fn prepend_to_content(&self, content: ClaudeContent, prefix: &str) -> ClaudeContent {
        match content {
            ClaudeContent::Text(text) => ClaudeContent::Text(format!("{}\n\n{}", prefix, text)),
            ClaudeContent::Blocks(mut blocks) => {
                // Prepend as a text block at the beginning
                blocks.insert(
                    0,
                    ContentBlock::Text {
                        text: prefix.to_string(),
                        cache_control: None,
                    },
                );
                ClaudeContent::Blocks(blocks)
            }
        }
    }

    fn convert_user_content(&self, user: &UserPromptPart) -> ClaudeContent {
        match &user.content {
            UserContent::Text(text) => ClaudeContent::Text(text.clone()),
            UserContent::Parts(parts) => {
                let blocks: Vec<ContentBlock> = parts
                    .iter()
                    .filter_map(|part| match part {
                        UserContentPart::Text { text } => Some(ContentBlock::Text {
                            text: text.clone(),
                            cache_control: None,
                        }),
                        UserContentPart::Image { image } => {
                            // Claude expects base64 image data
                            match image {
                                ImageContent::Binary(b) => Some(ContentBlock::Image {
                                    source: ImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: b.media_type.mime_type().to_string(),
                                        data: base64::engine::general_purpose::STANDARD
                                            .encode(&b.data),
                                    },
                                }),
                                ImageContent::Url(u) => {
                                    // Parse data URL if it's a base64 data URL
                                    if u.url.starts_with("data:") {
                                        let parts: Vec<&str> = u.url.splitn(2, ',').collect();
                                        if parts.len() == 2 {
                                            let media_type = parts[0]
                                                .strip_prefix("data:")
                                                .and_then(|s| s.strip_suffix(";base64"))
                                                .unwrap_or("image/png");
                                            Some(ContentBlock::Image {
                                                source: ImageSource {
                                                    source_type: "base64".to_string(),
                                                    media_type: media_type.to_string(),
                                                    data: parts[1].to_string(),
                                                },
                                            })
                                        } else {
                                            None
                                        }
                                    } else {
                                        None // URLs not directly supported
                                    }
                                }
                            }
                        }
                        _ => None,
                    })
                    .collect();
                ClaudeContent::Blocks(blocks)
            }
        }
    }

    fn convert_tools(&self, tools: &[serdes_ai_tools::ToolDefinition]) -> Vec<ClaudeTool> {
        tools
            .iter()
            .map(|t| ClaudeTool {
                name: format!("{}{}", TOOL_PREFIX, t.name),
                description: t.description.clone(),
                input_schema: t.parameters_json_schema.clone(),
            })
            .collect()
    }

    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> ClaudeRequest {
        let (system, mut api_messages) = self.convert_messages(messages);

        // Inject cache_control on the last content block of the last message
        // This matches the Python ClaudeCacheAsyncClient behavior
        if let Some(last_msg) = api_messages.last_mut() {
            if let ClaudeContent::Blocks(ref mut blocks) = last_msg.content {
                if let Some(last_block) = blocks.last_mut() {
                    if let ContentBlock::Text { cache_control, .. } = last_block {
                        *cache_control = Some(CacheControl::ephemeral());
                    }
                }
            }
        }

        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools(&params.tools))
        };

        // Build thinking config if enabled
        let thinking = if self.enable_thinking {
            Some(match self.thinking_budget {
                Some(budget) => ThinkingConfig::with_budget(budget),
                None => ThinkingConfig::enabled(),
            })
        } else {
            None
        };

        ClaudeRequest {
            model: self.model_name.clone(),
            messages: api_messages,
            // Use profile max_tokens as default, or 16384 if not set
            max_tokens: settings
                .max_tokens
                .map(|t| t as u32)
                .or(self.profile.max_tokens.map(|t| t as u32))
                .unwrap_or(16384),
            system,
            temperature: settings.temperature.map(|t| t as f32),
            stream: Some(stream),
            tools,
            tool_choice: params.tool_choice.as_ref().map(|tc| match tc {
                ToolChoice::Auto => serde_json::json!({"type": "auto"}),
                ToolChoice::Required => serde_json::json!({"type": "any"}),
                ToolChoice::None => serde_json::json!({"type": "none"}),
                ToolChoice::Specific(name) => serde_json::json!({
                    "type": "tool",
                    "name": name
                }),
            }),
            thinking,
        }
    }

    fn convert_response(&self, response: ClaudeResponse) -> ModelResponse {
        let mut parts = Vec::new();

        for block in &response.content {
            match block {
                ResponseContentBlock::Text { text } => {
                    parts.push(ModelResponsePart::Text(TextPart::new(text)));
                }
                ResponseContentBlock::ToolUse { id, name, input } => {
                    // Strip the cp_ prefix from tool names in responses
                    let unprefixed_name = name.strip_prefix(TOOL_PREFIX).unwrap_or(name);
                    parts.push(ModelResponsePart::ToolCall(
                        ToolCallPart::new(unprefixed_name, ToolCallArgs::Json(input.clone()))
                            .with_tool_call_id(id),
                    ));
                }
                ResponseContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    let mut thinking_part = ThinkingPart::new(thinking);
                    if let Some(sig) = signature {
                        thinking_part = thinking_part.with_signature(sig);
                    }
                    parts.push(ModelResponsePart::Thinking(thinking_part));
                }
                ResponseContentBlock::RedactedThinking { data } => {
                    parts.push(ModelResponsePart::Thinking(ThinkingPart::redacted(
                        data,
                        "anthropic",
                    )));
                }
            }
        }

        let finish_reason = response.stop_reason.as_ref().map(|r| match r.as_str() {
            "end_turn" | "stop" => FinishReason::Stop,
            "max_tokens" => FinishReason::Length,
            "tool_use" => FinishReason::ToolCall,
            _ => FinishReason::Stop,
        });

        let usage = response.usage.map(|u| RequestUsage {
            request_tokens: Some(u.input_tokens as u64),
            response_tokens: Some(u.output_tokens as u64),
            total_tokens: Some((u.input_tokens + u.output_tokens) as u64),
            cache_creation_tokens: None,
            cache_read_tokens: None,
            details: None,
        });

        ModelResponse {
            parts,
            model_name: Some(response.model),
            timestamp: chrono::Utc::now(),
            finish_reason,
            usage,
            vendor_id: Some(response.id),
            vendor_details: None,
            kind: "response".to_string(),
        }
    }

    /// Add an assistant response to messages (for multi-turn conversations).
    /// This is CRITICAL - Anthropic requires alternating user/assistant messages.
    pub fn add_response_to_messages(
        &self,
        messages: &mut Vec<ClaudeMessage>,
        response: &ModelResponse,
    ) {
        let mut blocks = Vec::new();

        for part in &response.parts {
            match part {
                ModelResponsePart::Text(text) => {
                    blocks.push(ContentBlock::Text {
                        text: text.content.clone(),
                        cache_control: None,
                    });
                }
                ModelResponsePart::ToolCall(tc) => {
                    // Re-add the cp_ prefix when building messages for Claude
                    let prefixed_name = if tc.tool_name.starts_with(TOOL_PREFIX) {
                        tc.tool_name.clone()
                    } else {
                        format!("{}{}", TOOL_PREFIX, tc.tool_name)
                    };
                    blocks.push(ContentBlock::ToolUse {
                        id: tc.tool_call_id.clone().unwrap_or_default(),
                        name: prefixed_name,
                        input: tc.args.to_json(),
                    });
                }
                ModelResponsePart::Thinking(think) => {
                    if think.is_redacted() {
                        if let Some(sig) = &think.signature {
                            blocks.push(ContentBlock::RedactedThinking { data: sig.clone() });
                        }
                    } else {
                        blocks.push(ContentBlock::Thinking {
                            thinking: think.content.clone(),
                            signature: think.signature.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        if !blocks.is_empty() {
            messages.push(ClaudeMessage {
                role: "assistant".to_string(),
                content: ClaudeContent::Blocks(blocks),
            });
        }
    }
}

#[async_trait]
impl Model for ClaudeCodeOAuthModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "claude-code-oauth"
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        use reqwest::header::{
            HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT,
        };

        let request_body = self.build_request(messages, settings, params, false);
        let url = format!("{}/v1/messages?beta=true", self.config.api_base_url);

        info!(
            model = %self.model_name,
            url = %url,
            message_count = messages.len(),
            "ClaudeCodeOAuth: making request"
        );
        debug!("ClaudeCodeOAuth: request body model={}", request_body.model);

        // Build headers for THIS request
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.access_token))
                .map_err(|e| ModelError::api(format!("Invalid auth header: {}", e)))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_str(&self.config.anthropic_version)
                .map_err(|e| ModelError::api(format!("Invalid version header: {}", e)))?,
        );
        headers.insert(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_static("oauth-2025-04-20,interleaved-thinking-2025-05-14"),
        );
        headers.insert(
            HeaderName::from_static("x-app"),
            HeaderValue::from_static("cli"),
        );
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("claude-cli/2.0.61 (external, cli)"),
        );

        debug!("ClaudeCodeOAuth: sending HTTP request...");
        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        debug!(status = %status, "ClaudeCodeOAuth: received response");

        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response.text().await.unwrap_or_default();
            error!(
                status = status_code,
                body_preview = %body.chars().take(500).collect::<String>(),
                "ClaudeCodeOAuth: API error"
            );

            return Err(ModelError::Http {
                status: status_code,
                body,
                headers: std::collections::HashMap::new(),
            });
        }

        let claude_response: ClaudeResponse = response.json().await?;
        info!(
            content_parts = claude_response.content.len(),
            "ClaudeCodeOAuth: parsed response successfully"
        );
        Ok(self.convert_response(claude_response))
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        use super::stream::ClaudeCodeStreamParser;
        use reqwest::header::{
            HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT,
        };

        // Build request with stream: true
        let request_body = self.build_request(messages, settings, params, true);
        let url = format!("{}/v1/messages?beta=true", self.config.api_base_url);

        info!(
            model = %self.model_name,
            url = %url,
            message_count = messages.len(),
            "ClaudeCodeOAuth: making streaming request"
        );

        // Build headers
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.access_token))
                .map_err(|e| ModelError::api(format!("Invalid auth header: {}", e)))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_str(&self.config.anthropic_version)
                .map_err(|e| ModelError::api(format!("Invalid version header: {}", e)))?,
        );
        headers.insert(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_static("oauth-2025-04-20,interleaved-thinking-2025-05-14"),
        );
        headers.insert(
            HeaderName::from_static("x-app"),
            HeaderValue::from_static("cli"),
        );
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("claude-cli/2.0.61 (external, cli)"),
        );

        debug!("ClaudeCodeOAuth: sending streaming HTTP request...");
        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        debug!(status = %status, "ClaudeCodeOAuth: received response");

        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response.text().await.unwrap_or_default();
            error!(
                status = status_code,
                body_preview = %body.chars().take(500).collect::<String>(),
                "ClaudeCodeOAuth: streaming API error"
            );

            return Err(ModelError::Http {
                status: status_code,
                body,
                headers: std::collections::HashMap::new(),
            });
        }

        info!("ClaudeCodeOAuth: streaming response started, creating parser");

        // Get the byte stream and wrap with SSE parser
        let byte_stream = response.bytes_stream();
        let parser = ClaudeCodeStreamParser::new(byte_stream);

        Ok(Box::pin(parser))
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }
}
