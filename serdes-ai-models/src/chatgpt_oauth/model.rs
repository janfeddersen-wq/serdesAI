//! ChatGPT OAuth model implementation.

use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::{openai_gpt4o_profile, ModelProfile};
use async_trait::async_trait;
use reqwest::Client;
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, 
    ModelSettings, RequestUsage,
};
use serdes_ai_core::messages::{
    ImageContent, PartStartEvent, TextPart, ToolCallArgs, ToolCallPart, 
    UserContent, UserContentPart, UserPromptPart,
};
use base64::Engine;
use std::time::Duration;

/// ChatGPT OAuth model.
///
/// Uses OAuth access tokens to authenticate with the ChatGPT Codex API.
/// This is an OpenAI-compatible API but with a different endpoint.
#[derive(Debug, Clone)]
pub struct ChatGptOAuthModel {
    model_name: String,
    access_token: String,
    client: Client,
    config: ChatGptConfig,
    profile: ModelProfile,
}

impl ChatGptOAuthModel {
    /// Create a new ChatGPT OAuth model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model name (e.g., "chatgpt-4o-codex")
    /// * `access_token` - OAuth access token from authentication flow
    pub fn new(model_name: impl Into<String>, access_token: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let profile = Self::profile_for_model(&model_name);
        
        Self {
            model_name,
            access_token: access_token.into(),
            client: Client::new(),
            config: ChatGptConfig::default(),
            profile,
        }
    }

    /// Set a custom config.
    #[must_use]
    pub fn with_config(mut self, config: ChatGptConfig) -> Self {
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

    /// Get the appropriate profile for a model name.
    fn profile_for_model(model: &str) -> ModelProfile {
        // ChatGPT Codex models are GPT-4o based
        let mut profile = openai_gpt4o_profile();
        
        if model.contains("o1") || model.contains("o3") {
            // Reasoning models
            profile.supports_reasoning = true;
        }
        
        profile
    }

    /// Convert our messages to Codex format.
    fn convert_messages(&self, requests: &[ModelRequest]) -> Vec<CodexMessage> {
        requests
            .iter()
            .flat_map(|req| self.convert_request(req))
            .collect()
    }

    fn convert_request(&self, req: &ModelRequest) -> Vec<CodexMessage> {
        let mut messages = Vec::new();

        for part in &req.parts {
            match part {
                ModelRequestPart::SystemPrompt(sys) => {
                    messages.push(CodexMessage {
                        role: "system".to_string(),
                        content: MessageContent::Text(sys.content.clone()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                ModelRequestPart::UserPrompt(user) => {
                    let content = self.convert_user_content(user);
                    messages.push(CodexMessage {
                        role: "user".to_string(),
                        content,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                ModelRequestPart::ToolReturn(ret) => {
                    messages.push(CodexMessage {
                        role: "tool".to_string(),
                        content: MessageContent::Text(ret.content.to_string_content()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: ret.tool_call_id.clone(),
                    });
                }
                ModelRequestPart::RetryPrompt(retry) => {
                    messages.push(CodexMessage {
                        role: "user".to_string(),
                        content: MessageContent::Text(retry.content.message().to_string()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                _ => {}
            }
        }

        messages
    }

    fn convert_user_content(&self, user: &UserPromptPart) -> MessageContent {
        match &user.content {
            UserContent::Text(text) => MessageContent::Text(text.clone()),
            UserContent::Parts(parts) => {
                let converted: Vec<ContentPart> = parts
                    .iter()
                    .filter_map(|part| match part {
                        UserContentPart::Text { text } => {
                            Some(ContentPart::Text { text: text.clone() })
                        }
                        UserContentPart::Image { image } => {
                            let url = match image {
                                ImageContent::Url(u) => u.url.clone(),
                                ImageContent::Binary(b) => {
                                    format!(
                                        "data:{};base64,{}",
                                        b.media_type.mime_type(),
                                        base64::engine::general_purpose::STANDARD.encode(&b.data)
                                    )
                                }
                            };
                            Some(ContentPart::ImageUrl {
                                image_url: ImageUrl { url, detail: None },
                            })
                        }
                        _ => None,
                    })
                    .collect();
                MessageContent::Parts(converted)
            }
        }
    }

    fn convert_tools(&self, tools: &[serdes_ai_tools::ToolDefinition]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters_json_schema
                    }
                })
            })
            .collect()
    }

    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> CodexRequest {
        let messages = self.convert_messages(messages);
        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools(&params.tools))
        };

        CodexRequest {
            model: self.model_name.clone(),
            messages,
            temperature: settings.temperature.map(|t| t as f32),
            max_tokens: settings.max_tokens.map(|t| t as u32),
            stream: Some(stream),
            tools,
            tool_choice: params.tool_choice.as_ref().map(|tc| match tc {
                ToolChoice::Auto => serde_json::json!("auto"),
                ToolChoice::Required => serde_json::json!("required"),
                ToolChoice::None => serde_json::json!("none"),
                ToolChoice::Specific(name) => serde_json::json!({
                    "type": "function",
                    "function": {"name": name}
                }),
            }),
        }
    }

    fn convert_response(&self, response: CodexResponse) -> ModelResponse {
        let mut parts = Vec::new();

        for choice in &response.choices {
            if let Some(content) = &choice.message.content {
                if !content.is_empty() {
                    parts.push(ModelResponsePart::Text(TextPart::new(content)));
                }
            }

            if let Some(tool_calls) = &choice.message.tool_calls {
                for tc in tool_calls {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    parts.push(ModelResponsePart::ToolCall(
                        ToolCallPart::new(&tc.function.name, ToolCallArgs::Json(args))
                            .with_tool_call_id(&tc.id),
                    ));
                }
            }
        }

        let finish_reason = response
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .map(|r| match r.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCall,
                _ => FinishReason::Stop,
            });

        let usage = response.usage.map(|u| RequestUsage {
            request_tokens: Some(u.prompt_tokens as u64),
            response_tokens: Some(u.completion_tokens as u64),
            total_tokens: Some(u.total_tokens as u64),
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
impl Model for ChatGptOAuthModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "chatgpt-oauth"
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let request = self.build_request(messages, settings, params, false);
        let url = format!("{}/chat/completions", self.config.api_base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
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

        let codex_response: CodexResponse = response.json().await?;
        Ok(self.convert_response(codex_response))
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
