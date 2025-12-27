//! Mistral AI model implementation.
//!
//! [Mistral AI](https://mistral.ai) provides powerful open-weight and commercial
//! language models with excellent instruction-following capabilities.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_models::mistral::MistralModel;
//!
//! let model = MistralModel::from_env("mistral-large-latest")?;
//! ```
//!
//! ## Available Models
//!
//! - `mistral-large-latest` - Most capable model, 128K context
//! - `mistral-medium-latest` - Balanced performance
//! - `mistral-small-latest` - Fast, efficient model
//! - `open-mixtral-8x22b` - Open-weight MoE model
//! - `open-mixtral-8x7b` - Efficient MoE model
//! - `open-mistral-7b` - Small open-weight model
//! - `codestral-latest` - Optimized for code

pub mod types;

use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::ModelProfile;
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
    ModelSettings, RequestUsage, TextPart, ToolCallPart, UserContent, UserContentPart,
};

/// Mistral AI model client.
#[derive(Debug, Clone)]
pub struct MistralModel {
    /// Model name.
    model_name: String,
    /// HTTP client.
    client: Client,
    /// API key.
    api_key: String,
    /// Base URL.
    base_url: String,
    /// Model profile.
    profile: ModelProfile,
    /// Default timeout.
    default_timeout: Duration,
}

impl MistralModel {
    /// Default Mistral API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.mistral.ai/v1";

    /// Create a new Mistral model.
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            client: Client::new(),
            api_key: api_key.into(),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            profile: Self::default_profile(),
            default_timeout: Duration::from_secs(120),
        }
    }

    /// Create from environment variable `MISTRAL_API_KEY`.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| ModelError::configuration("MISTRAL_API_KEY not set"))?;
        Ok(Self::new(model_name, api_key))
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set a custom profile.
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the default timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Default profile for Mistral models.
    fn default_profile() -> ModelProfile {
        ModelProfile {
            supports_tools: true,
            supports_parallel_tools: true,
            supports_native_structured_output: true,
            supports_strict_tools: false,
            supports_system_messages: true,
            supports_images: true,
            supports_streaming: true,
            ..Default::default()
        }
    }

    /// Build the chat request.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> types::ChatRequest {
        let api_messages = self.convert_messages(messages);
        let tools = self.convert_tools(params);

        types::ChatRequest {
            model: self.model_name.clone(),
            messages: api_messages,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice: None,
            temperature: settings.temperature.map(|t| t as f64),
            max_tokens: settings.max_tokens.map(|t| t as u64),
            top_p: settings.top_p.map(|t| t as f64),
            stream: false,
            safe_prompt: None,
            random_seed: settings.seed.map(|s| s as i64),
        }
    }

    /// Convert messages to Mistral format.
    fn convert_messages(&self, messages: &[ModelRequest]) -> Vec<types::Message> {
        let mut result = Vec::new();

        for request in messages {
            for part in &request.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sp) => {
                        result.push(types::Message {
                            role: types::Role::System,
                            content: types::Content::Text(sp.content.clone()),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    ModelRequestPart::UserPrompt(up) => {
                        let content = match &up.content {
                            UserContent::Text(t) => types::Content::Text(t.clone()),
                            UserContent::Parts(parts) => {
                                let text = parts
                                    .iter()
                                    .filter_map(|p| match p {
                                        UserContentPart::Text { text } => Some(text.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                types::Content::Text(text)
                            }
                        };
                        result.push(types::Message {
                            role: types::Role::User,
                            content,
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    ModelRequestPart::ToolReturn(tr) => {
                        result.push(types::Message {
                            role: types::Role::Tool,
                            content: types::Content::Text(tr.content.to_string_content()),
                            tool_calls: None,
                            tool_call_id: tr.tool_call_id.clone(),
                            name: Some(tr.tool_name.clone()),
                        });
                    }
                    ModelRequestPart::RetryPrompt(rp) => {
                        result.push(types::Message {
                            role: types::Role::User,
                            content: types::Content::Text(rp.content.message().to_string()),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        });
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        result.push(types::Message {
                            role: types::Role::Tool,
                            content: types::Content::Text(content_str),
                            tool_calls: None,
                            tool_call_id: Some(builtin.tool_call_id.clone()),
                            name: Some(builtin.tool_name.clone()),
                        });
                    }
                }
            }
        }

        result
    }

    /// Convert tools to Mistral format.
    fn convert_tools(&self, params: &ModelRequestParameters) -> Vec<types::Tool> {
        params
            .tools
            .iter()
            .map(|t| types::Tool {
                r#type: "function".to_string(),
                function: types::FunctionDef {
                    name: t.name.clone(),
                    description: Some(t.description.clone()),
                    parameters: serde_json::to_value(&t.parameters_json_schema)
                        .unwrap_or_default(),
                },
            })
            .collect()
    }

    /// Parse response from Mistral.
    fn parse_response(&self, response: types::ChatResponse) -> Result<ModelResponse, ModelError> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::invalid_response("No choices in response"))?;

        let mut parts = Vec::new();

        // Text content
        if let types::Content::Text(text) = choice.message.content {
            if !text.is_empty() {
                parts.push(ModelResponsePart::Text(TextPart::new(text)));
            }
        }

        // Tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let args: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                    tool_name: tc.function.name,
                    args: serdes_ai_core::messages::ToolCallArgs::Json(args),
                    tool_call_id: Some(tc.id),
                    id: None,
                    provider_details: None,
                }));
            }
        }

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => Some(FinishReason::EndTurn),
            Some("length") => Some(FinishReason::Length),
            Some("tool_calls") => Some(FinishReason::ToolCall),
            _ => None,
        };

        let usage = response.usage.map(|u| RequestUsage {
            request_tokens: Some(u.prompt_tokens as u64),
            response_tokens: Some(u.completion_tokens as u64),
            total_tokens: Some(u.total_tokens as u64),
            ..Default::default()
        });

        Ok(ModelResponse {
            parts,
            finish_reason,
            usage,
            model_name: Some(response.model),
            timestamp: serdes_ai_core::identifier::now_utc(),
            vendor_id: Some(response.id),
            vendor_details: None,
            kind: "response".to_string(),
        })
    }
}

#[async_trait]
impl Model for MistralModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "mistral"
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let body = self.build_request(messages, settings, params);
        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(timeout)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(ModelError::http(status, text));
        }

        let chat_response: types::ChatResponse = response
            .json()
            .await
            .map_err(|e| ModelError::invalid_response(e.to_string()))?;

        self.parse_response(chat_response)
    }

    async fn request_stream(
        &self,
        _messages: &[ModelRequest],
        _settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        Err(ModelError::not_supported("Streaming for Mistral"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_model_creation() {
        let model = MistralModel::new("mistral-large-latest", "test-key");
        assert_eq!(model.name(), "mistral-large-latest");
        assert_eq!(model.system(), "mistral");
    }

    #[test]
    fn test_mistral_with_settings() {
        let model = MistralModel::new("mistral-small-latest", "key")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(model.default_timeout, Duration::from_secs(60));
    }
}
