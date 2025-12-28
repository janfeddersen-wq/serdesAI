//! Claude Code OAuth model implementation.

use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::{anthropic_claude_profile, ModelProfile};
use async_trait::async_trait;
use reqwest::Client;
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, 
    ModelSettings, RequestUsage,
};
use serdes_ai_core::messages::{
    ImageContent, PartStartEvent, TextPart, ThinkingPart, ToolCallArgs, ToolCallPart, 
    UserContent, UserContentPart, UserPromptPart,
};
use base64::Engine;
use std::time::Duration;

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
        let profile = anthropic_claude_profile();
        
        Self {
            model_name,
            access_token: access_token.into(),
            client: Client::new(),
            config: ClaudeCodeConfig::default(),
            profile,
        }
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

    /// Convert our messages to Claude format.
    fn convert_messages(&self, requests: &[ModelRequest]) -> (Option<String>, Vec<ClaudeMessage>) {
        let mut system_prompt = None;
        let mut messages = Vec::new();

        for req in requests {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        // Claude uses a separate system field
                        system_prompt = Some(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        let content = self.convert_user_content(user);
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
                    _ => {}
                }
            }
        }

        (system_prompt, messages)
    }

    fn convert_user_content(&self, user: &UserPromptPart) -> ClaudeContent {
        match &user.content {
            UserContent::Text(text) => ClaudeContent::Text(text.clone()),
            UserContent::Parts(parts) => {
                let blocks: Vec<ContentBlock> = parts
                    .iter()
                    .filter_map(|part| match part {
                        UserContentPart::Text { text } => {
                            Some(ContentBlock::Text { text: text.clone() })
                        }
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
                name: t.name.clone(),
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
        let (system, messages) = self.convert_messages(messages);
        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools(&params.tools))
        };

        ClaudeRequest {
            model: self.model_name.clone(),
            messages,
            max_tokens: settings.max_tokens.map(|t| t as u32).unwrap_or(4096),
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
                    parts.push(ModelResponsePart::ToolCall(
                        ToolCallPart::new(name, ToolCallArgs::Json(input.clone()))
                            .with_tool_call_id(id),
                    ));
                }
                ResponseContentBlock::Thinking { thinking } => {
                    parts.push(ModelResponsePart::Thinking(ThinkingPart::new(thinking)));
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
        let request = self.build_request(messages, settings, params, false);
        let url = format!("{}/v1/messages", self.config.api_base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .header("anthropic-version", &self.config.anthropic_version)
            .header("x-api-key", &self.access_token)  // Anthropic also accepts x-api-key
            .timeout(Duration::from_secs(120))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(ModelError::Http {
                status,
                body,
                headers: std::collections::HashMap::new(),
            });
        }

        let claude_response: ClaudeResponse = response.json().await?;
        Ok(self.convert_response(claude_response))
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        // For now, fall back to non-streaming
        // TODO: Implement proper SSE streaming
        let response = self.request(messages, settings, params).await?;
        
        use serdes_ai_core::messages::ModelResponseStreamEvent;
        
        let events: Vec<Result<ModelResponseStreamEvent, ModelError>> = response
            .parts
            .into_iter()
            .enumerate()
            .map(|(idx, part)| {
                Ok(ModelResponseStreamEvent::PartStart(PartStartEvent::new(idx, part)))
            })
            .collect();
        
        Ok(Box::pin(futures::stream::iter(events)))
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }
}
