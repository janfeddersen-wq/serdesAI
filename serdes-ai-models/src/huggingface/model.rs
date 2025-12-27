//! HuggingFace model implementation.

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use std::time::Duration;

use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::ModelProfile;
use serdes_ai_core::{
    messages::ModelResponseStreamEvent, FinishReason, ModelRequest, ModelRequestPart,
    ModelResponse, ModelResponsePart, ModelSettings, TextPart, UserContent, UserContentPart,
};

/// HuggingFace Inference API base URL.
pub const HF_INFERENCE_URL: &str = "https://api-inference.huggingface.co/models";

/// HuggingFace model client.
///
/// Supports both the HuggingFace Inference API and self-hosted TGI endpoints.
#[derive(Debug, Clone)]
pub struct HuggingFaceModel {
    /// Model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    model_id: String,
    /// API token.
    api_token: String,
    /// HTTP client.
    client: Client,
    /// Custom endpoint URL (for self-hosted TGI).
    endpoint: Option<String>,
    /// Model profile.
    profile: ModelProfile,
    /// Default timeout.
    default_timeout: Duration,
}

impl HuggingFaceModel {
    /// Create a new HuggingFace model.
    pub fn new(model_id: impl Into<String>, api_token: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_token: api_token.into(),
            client: Client::new(),
            endpoint: None,
            profile: Self::default_profile(),
            default_timeout: Duration::from_secs(120),
        }
    }

    /// Create from environment variable `HF_TOKEN` or `HUGGINGFACE_API_TOKEN`.
    pub fn from_env(model_id: impl Into<String>) -> Result<Self, ModelError> {
        let api_token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGINGFACE_API_TOKEN"))
            .map_err(|_| ModelError::configuration("HF_TOKEN or HUGGINGFACE_API_TOKEN not set"))?;
        Ok(Self::new(model_id, api_token))
    }

    /// Set a custom endpoint URL (for self-hosted TGI).
    #[must_use]
    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = Some(url.into());
        self
    }

    /// Set a custom profile.
    #[must_use]
    pub fn with_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set default timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Get the API URL for this model.
    fn api_url(&self) -> String {
        match &self.endpoint {
            Some(url) => url.clone(),
            None => format!("{}/{}", HF_INFERENCE_URL, self.model_id),
        }
    }

    /// Default profile for HuggingFace models.
    fn default_profile() -> ModelProfile {
        ModelProfile {
            supports_tools: false,
            supports_parallel_tools: false,
            supports_native_structured_output: false,
            supports_strict_tools: false,
            supports_system_messages: true,
            supports_images: false,
            supports_streaming: true,
            ..Default::default()
        }
    }

    /// Convert messages to a single prompt string.
    fn build_prompt(&self, messages: &[ModelRequest]) -> String {
        let mut prompt = String::new();

        for request in messages {
            for part in &request.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sp) => {
                        prompt.push_str(&format!("<|system|>\n{}\n", sp.content));
                    }
                    ModelRequestPart::UserPrompt(up) => {
                        let text = match &up.content {
                            UserContent::Text(t) => t.clone(),
                            UserContent::Parts(parts) => parts
                                .iter()
                                .filter_map(|p| match p {
                                    UserContentPart::Text { text } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        prompt.push_str(&format!("<|user|>\n{}\n", text));
                    }
                    ModelRequestPart::ToolReturn(tr) => {
                        prompt.push_str(&format!("<|tool|>\n{}\n", tr.content.to_string_content()));
                    }
                    ModelRequestPart::RetryPrompt(rp) => {
                        prompt.push_str(&format!("<|user|>\n{}\n", rp.content.message()));
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        prompt.push_str(&format!("<|tool|>\n{}\n", content_str));
                    }
                }
            }
        }

        prompt.push_str("<|assistant|>\n");
        prompt
    }

    /// Build generation parameters from settings.
    fn build_parameters(&self, settings: &ModelSettings) -> GenerateParameters {
        let mut params = GenerateParameters::new();

        if let Some(temp) = settings.temperature {
            params = params.temperature(temp as f32);
        }
        if let Some(top_p) = settings.top_p {
            params.top_p = Some(top_p as f32);
        }
        if let Some(max) = settings.max_tokens {
            params.max_new_tokens = Some(max as u32);
        }
        if let Some(stop) = &settings.stop {
            params.stop = Some(stop.clone());
        }
        if let Some(seed) = settings.seed {
            params.seed = Some(seed as u64);
        }

        // Don't return the prompt in the output
        params.return_full_text = Some(false);
        params.details = Some(true);

        params
    }

    /// Parse generation response into ModelResponse.
    fn parse_response(&self, resp: GenerateResponse) -> Result<ModelResponse, ModelError> {
        let text = resp
            .text()
            .ok_or_else(|| ModelError::invalid_response("No generated text"))?;

        let finish_reason = match &resp {
            GenerateResponse::Single(r) => r.details.as_ref().and_then(|d| {
                d.finish_reason.as_ref().map(|r| match r.as_str() {
                    "length" => FinishReason::Length,
                    "eos_token" | "stop" => FinishReason::Stop,
                    _ => FinishReason::Stop,
                })
            }),
            GenerateResponse::Batch(results) => results.first().and_then(|r| {
                r.details.as_ref().and_then(|d| {
                    d.finish_reason.as_ref().map(|r| match r.as_str() {
                        "length" => FinishReason::Length,
                        "eos_token" | "stop" => FinishReason::Stop,
                        _ => FinishReason::Stop,
                    })
                })
            }),
        };

        Ok(ModelResponse {
            parts: vec![ModelResponsePart::Text(TextPart::new(text.to_string()))],
            finish_reason,
            usage: None, // TGI doesn't provide token counts in basic mode
            model_name: Some(self.model_id.clone()),
            timestamp: serdes_ai_core::identifier::now_utc(),
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        })
    }

    /// Handle API error response.
    fn handle_error(&self, status: u16, body: &str) -> ModelError {
        if let Ok(err) = serde_json::from_str::<HuggingFaceError>(body) {
            match status {
                401 | 403 => ModelError::auth(err.error),
                429 => ModelError::rate_limited(None),
                404 => ModelError::NotFound(err.error),
                503 => ModelError::api(format!("Model loading: {}", err.error)),
                _ => ModelError::Api {
                    message: err.error,
                    code: err.error_type,
                },
            }
        } else {
            ModelError::http(status, body)
        }
    }
}

#[async_trait]
impl Model for HuggingFaceModel {
    fn name(&self) -> &str {
        &self.model_id
    }

    fn system(&self) -> &str {
        "huggingface"
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let prompt = self.build_prompt(messages);
        let parameters = self.build_parameters(settings);

        let request_body = GenerateRequest::new(prompt).with_parameters(parameters);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", self.api_token))
            .header("Content-Type", "application/json")
            .timeout(timeout)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error(status, &body));
        }

        let resp: GenerateResponse = response
            .json()
            .await
            .map_err(|e| ModelError::invalid_response(e.to_string()))?;

        self.parse_response(resp)
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        let prompt = self.build_prompt(messages);
        let parameters = self.build_parameters(settings);

        let request_body = GenerateRequest::new(prompt)
            .with_parameters(parameters)
            .with_stream(true);

        let timeout = settings.timeout.unwrap_or(self.default_timeout);

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", self.api_token))
            .header("Content-Type", "application/json")
            .timeout(timeout)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error(status, &body));
        }

        let stream = response.bytes_stream();
        let model_id = self.model_id.clone();

        // Parse SSE stream
        let mapped = stream.filter_map(move |chunk| {
            let _model_id = model_id.clone();
            async move {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        // Parse SSE data lines
                        for line in text.lines() {
                            if let Some(data) = line.strip_prefix("data:") {
                                let data = data.trim();
                                if data.is_empty() || data == "[DONE]" {
                                    continue;
                                }
                                if let Ok(resp) = serde_json::from_str::<StreamResponse>(data) {
                                    if let Some(token) = resp.token {
                                        if !token.special {
                                            return Some(Ok(
                                                ModelResponseStreamEvent::text_delta(0, token.text),
                                            ));
                                        }
                                    }
                                    if resp.generated_text.is_some() {
                                        return Some(Ok(
                                            ModelResponseStreamEvent::part_end(0),
                                        ));
                                    }
                                }
                            }
                        }
                        None
                    }
                    Err(e) => Some(Err(ModelError::network(e.to_string()))),
                }
            }
        });

        Ok(Box::pin(mapped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_model_creation() {
        let model = HuggingFaceModel::new("meta-llama/Llama-3.1-8B-Instruct", "test-token");
        assert_eq!(model.name(), "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(model.system(), "huggingface");
    }

    #[test]
    fn test_custom_endpoint() {
        let model = HuggingFaceModel::new("my-model", "token")
            .with_endpoint("http://localhost:8080/generate");
        assert_eq!(model.api_url(), "http://localhost:8080/generate");
    }

    #[test]
    fn test_default_api_url() {
        let model = HuggingFaceModel::new("meta-llama/Llama-3.1-8B", "token");
        assert_eq!(
            model.api_url(),
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B"
        );
    }

    #[test]
    fn test_prompt_building() {
        let model = HuggingFaceModel::new("test", "token");
        let mut req = ModelRequest::new();
        req.add_user_prompt("Hello!");

        let prompt = model.build_prompt(&[req]);
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }
}
