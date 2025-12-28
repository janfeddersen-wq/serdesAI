//! Ollama model implementation for local models.
//!
//! [Ollama](https://ollama.ai) allows running LLMs locally. This implementation
//! connects to a local Ollama server.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_models::ollama::OllamaModel;
//!
//! // Connect to local Ollama
//! let model = OllamaModel::new("llama3.1");
//!
//! // Custom host
//! let model = OllamaModel::new("codellama")
//!     .with_base_url("http://192.168.1.100:11434");
//! ```
//!
//! ## Available Models
//!
//! Run `ollama list` to see installed models. Popular options:
//! - `llama3.1` - Meta's Llama 3.1
//! - `llama3.1:70b` - Larger Llama 3.1
//! - `mistral` - Mistral 7B
//! - `mixtral` - Mixtral 8x7B
//! - `codellama` - Code-focused model
//! - `phi3` - Microsoft Phi-3
//! - `gemma2` - Google Gemma 2
//! - `qwen2` - Alibaba Qwen 2

pub mod types;

use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::ModelProfile;
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
    ModelSettings, RequestUsage, TextPart, ToolCallPart, UserContent, UserContentPart,
};
use serdes_ai_core::messages::ImageContent;

/// Ollama model client.
#[derive(Debug, Clone)]
pub struct OllamaModel {
    /// Model name.
    model_name: String,
    /// HTTP client.
    client: Client,
    /// Base URL.
    base_url: String,
    /// Model profile.
    profile: ModelProfile,
    /// Keep alive duration.
    keep_alive: Option<String>,
    /// Default timeout.
    default_timeout: Duration,
}

impl OllamaModel {
    /// Default Ollama base URL.
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434";

    /// Create a new Ollama model.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            client: Client::new(),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            profile: Self::default_profile(),
            keep_alive: None,
            default_timeout: Duration::from_secs(300),
        }
    }

    /// Create from environment variable `OLLAMA_HOST`.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let base_url = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_string());
        Ok(Self::new(model_name).with_base_url(base_url))
    }

    /// Set custom base URL.
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

    /// Set keep alive duration (e.g., "5m", "1h").
    pub fn with_keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.keep_alive = Some(duration.into());
        self
    }

    /// Default profile for Ollama models.
    fn default_profile() -> ModelProfile {
        ModelProfile {
            supports_tools: true,
            supports_parallel_tools: false,
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
    ) -> Result<types::ChatRequest, ModelError> {
        let api_messages = self.convert_messages(messages)?;
        let tools = self.convert_tools(params);
        let tool_choice = params
            .tool_choice
            .as_ref()
            .map(|choice| self.convert_tool_choice(choice));

        let options = types::Options {
            temperature: settings.temperature.map(|t| t as f64),
            top_p: settings.top_p.map(|t| t as f64),
            top_k: None,
            num_predict: settings.max_tokens.map(|n| n as i32),
            stop: settings.stop.clone(),
            seed: settings.seed.map(|s| s as i64),
            num_ctx: None,
            repeat_penalty: None,
            repeat_last_n: None,
        };

        Ok(types::ChatRequest {
            model: self.model_name.clone(),
            messages: api_messages,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            stream: Some(false),
            options: Some(options),
            keep_alive: self.keep_alive.clone(),
            format: None,
        })
    }

    /// Convert messages to Ollama format.
    fn convert_messages(
        &self,
        messages: &[ModelRequest],
    ) -> Result<Vec<types::Message>, ModelError> {
        let mut result = Vec::new();

        for request in messages {
            for part in &request.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sp) => {
                        result.push(types::Message {
                            role: "system".to_string(),
                            content: sp.content.clone(),
                            images: None,
                            tool_calls: None,
                        });
                    }
                    ModelRequestPart::UserPrompt(up) => {
                        let (content, images) = self.convert_user_content(&up.content)?;
                        result.push(types::Message {
                            role: "user".to_string(),
                            content,
                            images,
                            tool_calls: None,
                        });
                    }
                    ModelRequestPart::ToolReturn(tr) => {
                        result.push(types::Message {
                            role: "tool".to_string(),
                            content: tr.content.to_string_content(),
                            images: None,
                            tool_calls: None,
                        });
                    }
                    ModelRequestPart::RetryPrompt(rp) => {
                        result.push(types::Message {
                            role: "user".to_string(),
                            content: rp.content.message().to_string(),
                            images: None,
                            tool_calls: None,
                        });
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        result.push(types::Message {
                            role: "tool".to_string(),
                            content: content_str,
                            images: None,
                            tool_calls: None,
                        });
                    }
                    ModelRequestPart::ModelResponse(response) => {
                        // Add assistant response for proper alternation
                        let mut text_content = String::new();
                        for resp_part in &response.parts {
                            if let serdes_ai_core::ModelResponsePart::Text(t) = resp_part {
                                text_content.push_str(&t.content);
                            }
                        }
                        result.push(types::Message {
                            role: "assistant".to_string(),
                            content: text_content,
                            images: None,
                            tool_calls: None,
                        });
                    }
                }
            }
        }

        Ok(result)
    }

    /// Convert user content, extracting images.
    fn convert_user_content(
        &self,
        content: &UserContent,
    ) -> Result<(String, Option<Vec<String>>), ModelError> {
        match content {
            UserContent::Text(t) => Ok((t.clone(), None)),
            UserContent::Parts(parts) => {
                let mut text = String::new();
                let mut images = Vec::new();

                for part in parts {
                    match part {
                        UserContentPart::Text { text: text_part } => {
                            text.push_str(text_part);
                        }
                        UserContentPart::Image { image } => {
                            if !self.profile.supports_images {
                                return Err(ModelError::unsupported_content("images"));
                            }

                            match image {
                                ImageContent::Binary(binary) => {
                                    use base64::Engine;
                                    let encoded = base64::engine::general_purpose::STANDARD
                                        .encode(&binary.data);
                                    images.push(encoded);
                                }
                                ImageContent::Url(_) => {
                                    return Err(ModelError::unsupported_content("image URLs"));
                                }
                            }
                        }
                        UserContentPart::Document { .. } => {
                            return Err(ModelError::unsupported_content("documents"));
                        }
                        _ => {
                            return Err(ModelError::unsupported_content("non-text content"));
                        }
                    }
                }

                let images = if images.is_empty() { None } else { Some(images) };
                Ok((text, images))
            }
        }
    }

    /// Convert tools to Ollama format.
    fn convert_tools(&self, params: &ModelRequestParameters) -> Vec<types::Tool> {
        params
            .tools
            .iter()
            .map(|t| types::Tool {
                r#type: "function".to_string(),
                function: types::FunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: serde_json::to_value(&t.parameters_json_schema)
                        .unwrap_or_default(),
                },
            })
            .collect()
    }

    /// Convert tool choice to Ollama format.
    fn convert_tool_choice(&self, choice: &ToolChoice) -> String {
        match choice {
            ToolChoice::Auto => "auto".to_string(),
            ToolChoice::Required => "required".to_string(),
            ToolChoice::None => "none".to_string(),
            ToolChoice::Specific(name) => name.clone(),
        }
    }

    /// Parse response from Ollama.
    fn parse_response(&self, response: types::ChatResponse) -> Result<ModelResponse, ModelError> {
        let mut parts = Vec::new();

        // Text content
        if !response.message.content.is_empty() {
            parts.push(ModelResponsePart::Text(TextPart::new(response.message.content)));
        }

        // Tool calls
        if let Some(tool_calls) = response.message.tool_calls {
            for (idx, tc) in tool_calls.into_iter().enumerate() {
                parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                    tool_name: tc.function.name,
                    args: serdes_ai_core::messages::ToolCallArgs::Json(tc.function.arguments),
                    tool_call_id: Some(format!("call_{}", idx)),
                    id: None,
                    provider_details: None,
                }));
            }
        }

        let finish_reason = if response.done {
            match response.done_reason.as_deref() {
                Some("stop") => Some(FinishReason::EndTurn),
                Some("length") => Some(FinishReason::Length),
                _ => Some(FinishReason::EndTurn),
            }
        } else {
            None
        };

        let usage = if response.prompt_eval_count.is_some() || response.eval_count.is_some() {
            Some(RequestUsage {
                request_tokens: response.prompt_eval_count.map(|n| n as u64),
                response_tokens: response.eval_count.map(|n| n as u64),
                total_tokens: None,
                ..Default::default()
            })
        } else {
            None
        };

        Ok(ModelResponse {
            parts,
            finish_reason,
            usage,
            model_name: Some(response.model),
            timestamp: serdes_ai_core::identifier::now_utc(),
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        })
    }
}

#[async_trait]
impl Model for OllamaModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "ollama"
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
        let body = self.build_request(messages, settings, params)?;
        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
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
        Err(ModelError::not_supported("Streaming for Ollama"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_model_creation() {
        let model = OllamaModel::new("llama3.1");
        assert_eq!(model.name(), "llama3.1");
        assert_eq!(model.system(), "ollama");
        assert_eq!(model.base_url, OllamaModel::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_ollama_custom_url() {
        let model = OllamaModel::new("mistral")
            .with_base_url("http://192.168.1.100:11434");
        assert_eq!(model.base_url, "http://192.168.1.100:11434");
    }

    #[test]
    fn test_ollama_with_keep_alive() {
        let model = OllamaModel::new("phi3")
            .with_keep_alive("10m");

        assert_eq!(model.keep_alive, Some("10m".to_string()));
    }
}
