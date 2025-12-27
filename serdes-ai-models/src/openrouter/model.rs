//! OpenRouter model - OpenAI-compatible API routing to multiple providers.

use super::types::{OpenRouterExtras, ProviderPreferences};
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::openai::{stream::OpenAIStreamParser, types::*};
use crate::profile::{openai_gpt4o_profile, ModelProfile};
use async_trait::async_trait;
use reqwest::Client;
use serdes_ai_core::{
    messages::{TextPart, ToolCallArgs, ToolCallPart, UserContent, UserContentPart},
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
    ModelSettings, RequestUsage,
};
use serdes_ai_tools::ToolDefinition;
use std::time::Duration;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter model - routes requests to multiple providers.
#[derive(Debug, Clone)]
pub struct OpenRouterModel {
    model_name: String,
    client: Client,
    api_key: String,
    base_url: String,
    http_referer: Option<String>,
    app_title: Option<String>,
    provider_preferences: Option<ProviderPreferences>,
    transforms: Option<Vec<String>>,
    profile: ModelProfile,
    default_timeout: Duration,
}

impl OpenRouterModel {
    /// Create a new OpenRouter model. Model names: `provider/model` (e.g., `anthropic/claude-3-opus`).
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let profile = Self::profile_for_model(&model_name);
        Self { model_name, client: Client::new(), api_key: api_key.into(),
            base_url: OPENROUTER_BASE_URL.into(), http_referer: None, app_title: None,
            provider_preferences: None, transforms: None, profile, default_timeout: Duration::from_secs(120) }
    }

    /// Create from environment variable `OPENROUTER_API_KEY`.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| ModelError::Configuration("OPENROUTER_API_KEY not set".into()))?;
        Ok(Self::new(model_name, api_key))
    }

    /// Set HTTP Referer header (recommended by OpenRouter).
    #[must_use]
    pub fn with_http_referer(mut self, r: impl Into<String>) -> Self { self.http_referer = Some(r.into()); self }
    /// Set X-Title header (app name).
    #[must_use]
    pub fn with_app_title(mut self, t: impl Into<String>) -> Self { self.app_title = Some(t.into()); self }
    /// Set provider routing preferences.
    #[must_use]
    pub fn with_provider_preferences(mut self, p: ProviderPreferences) -> Self { self.provider_preferences = Some(p); self }
    /// Set message transforms.
    #[must_use]
    pub fn with_transforms(mut self, t: Vec<String>) -> Self { self.transforms = Some(t); self }

    /// Determine profile based on model name.
    fn profile_for_model(model: &str) -> ModelProfile {
        // Check for reasoning models
        if model.contains("/o1") || model.contains("/o3") {
            return crate::profile::openai_o1_profile();
        }
        // Check for Claude models
        if model.starts_with("anthropic/") {
            return crate::profile::anthropic_claude_profile();
        }
        // Default to GPT-4o profile (good general-purpose profile)
        openai_gpt4o_profile()
    }

    /// Build OpenRouter extras for the request.
    fn build_extras(&self) -> Option<OpenRouterExtras> {
        if self.provider_preferences.is_none() && self.transforms.is_none() {
            return None;
        }

        Some(OpenRouterExtras {
            provider: self.provider_preferences.clone(),
            transforms: self.transforms.clone(),
            route: None,
        })
    }

    fn convert_messages(&self, requests: &[ModelRequest]) -> Vec<ChatMessage> {
        requests.iter().flat_map(|req| req.parts.iter().filter_map(|part| match part {
            ModelRequestPart::SystemPrompt(sys) => Some(ChatMessage::system(&sys.content)),
            ModelRequestPart::UserPrompt(user) => Some(ChatMessage {
                role: "user".into(), name: None, tool_calls: None, tool_call_id: None,
                content: Some(match &user.content {
                    UserContent::Text(t) => MessageContent::Text(t.clone()),
                    UserContent::Parts(parts) => MessageContent::Parts(
                        parts.iter().filter_map(|p| match p {
                            UserContentPart::Text { text } => Some(ContentPart::text(text)),
                            _ => None,
                        }).collect()
                    ),
                }),
            }),
            ModelRequestPart::ToolReturn(ret) => Some(ChatMessage::tool(
                ret.tool_call_id.clone().unwrap_or_default(), ret.content.to_string_content()
            )),
            ModelRequestPart::RetryPrompt(retry) => Some(ChatMessage::user(retry.content.message())),
            ModelRequestPart::BuiltinToolReturn(builtin) => {
                let content_str = serde_json::to_string(&builtin.content)
                    .unwrap_or_else(|_| builtin.content_type().to_string());
                Some(ChatMessage::tool(builtin.tool_call_id.clone(), content_str))
            }
        })).collect()
    }

    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<ChatTool> {
        tools.iter().map(|t| {
            let p = serde_json::to_value(&t.parameters_json_schema).unwrap_or(serde_json::json!({}));
            if t.strict.unwrap_or(false) { ChatTool::function_strict(&t.name, &t.description, p) }
            else { ChatTool::function(&t.name, &t.description, p) }
        }).collect()
    }

    fn convert_tool_choice(&self, choice: &ToolChoice) -> ToolChoiceValue {
        match choice {
            ToolChoice::Auto => ToolChoiceValue::auto(),
            ToolChoice::Required => ToolChoiceValue::required(),
            ToolChoice::None => ToolChoiceValue::none(),
            ToolChoice::Specific(name) => ToolChoiceValue::function(name),
        }
    }

    /// Build request body with OpenRouter extras.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> serde_json::Value {
        let messages = self.convert_messages(messages);
        let tools = if params.tools.is_empty() { None } else { Some(self.convert_tools(&params.tools)) };
        let tool_choice = params.tool_choice.as_ref().map(|c| self.convert_tool_choice(c));

        let response_format = params.output_schema.as_ref().map(|schema| {
            let schema_value = serde_json::to_value(schema).unwrap_or(serde_json::json!({}));
            ResponseFormat::json_schema("output", schema_value, true)
        });

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        let obj = body.as_object_mut().unwrap();
        if let Some(temp) = settings.temperature { obj.insert("temperature".into(), temp.into()); }
        if let Some(top_p) = settings.top_p { obj.insert("top_p".into(), top_p.into()); }
        if let Some(max) = settings.max_tokens { obj.insert("max_tokens".into(), max.into()); }
        if let Some(stop) = &settings.stop { obj.insert("stop".into(), serde_json::to_value(stop).unwrap()); }
        if let Some(tools) = tools { obj.insert("tools".into(), serde_json::to_value(tools).unwrap()); }
        if let Some(tc) = tool_choice { obj.insert("tool_choice".into(), serde_json::to_value(tc).unwrap()); }
        if let Some(rf) = response_format { obj.insert("response_format".into(), serde_json::to_value(rf).unwrap()); }
        if stream {
            obj.insert("stream".into(), true.into());
            obj.insert("stream_options".into(), serde_json::json!({ "include_usage": true }));
        }

        // Add OpenRouter extras
        if let Some(extras) = self.build_extras() {
            if let Some(provider) = &extras.provider {
                obj.insert("provider".into(), serde_json::to_value(provider).unwrap());
            }
            if let Some(transforms) = &extras.transforms {
                obj.insert("transforms".into(), serde_json::to_value(transforms).unwrap());
            }
        }

        body
    }

    fn parse_response(&self, resp: ChatCompletionResponse) -> Result<ModelResponse, ModelError> {
        let choice = resp.choices.into_iter().next().ok_or_else(|| ModelError::invalid_response("No choices"))?;
        if let Some(refusal) = choice.message.refusal { return Err(ModelError::ContentFiltered(refusal)); }

        let mut parts = Vec::new();
        if let Some(c) = choice.message.content { if !c.is_empty() { parts.push(ModelResponsePart::Text(TextPart::new(c))); } }
        if let Some(tcs) = choice.message.tool_calls {
            for tc in tcs {
                let args = serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
                parts.push(ModelResponsePart::ToolCall(ToolCallPart::new(tc.function.name, ToolCallArgs::Json(args)).with_tool_call_id(tc.id)));
            }
        }

        let finish_reason = choice.finish_reason.map(|r| match r.as_str() {
            "stop" => FinishReason::Stop, "length" => FinishReason::Length,
            "content_filter" => FinishReason::ContentFilter, "tool_calls" => FinishReason::ToolCall, _ => FinishReason::Stop,
        });
        let usage = resp.usage.map(|u| RequestUsage {
            request_tokens: Some(u.prompt_tokens), response_tokens: Some(u.completion_tokens),
            total_tokens: Some(u.total_tokens), cache_creation_tokens: None,
            cache_read_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens), details: None,
        });

        Ok(ModelResponse { parts, model_name: Some(resp.model), timestamp: chrono::Utc::now(),
            finish_reason, usage, vendor_id: Some(resp.id), vendor_details: None, kind: "response".into() })
    }

    fn handle_error(&self, status: u16, body: &str) -> ModelError {
        serde_json::from_str::<OpenAIError>(body).map(|e| match status {
            401 => ModelError::auth(e.error.message),
            429 => ModelError::rate_limited(None),
            404 => ModelError::NotFound(e.error.message),
            _ => ModelError::Api { message: e.error.message, code: e.error.code },
        }).unwrap_or_else(|_| ModelError::http(status, body))
    }
}

impl OpenRouterModel {
    /// Build and send HTTP request with OpenRouter headers.
    async fn send_request(&self, body: &serde_json::Value, timeout: Duration) -> Result<reqwest::Response, ModelError> {
        let mut req = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(timeout);

        if let Some(ref referer) = self.http_referer { req = req.header("HTTP-Referer", referer); }
        if let Some(ref title) = self.app_title { req = req.header("X-Title", title); }

        let response = req.json(body).send().await?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error(status, &body));
        }
        Ok(response)
    }
}

#[async_trait]
impl Model for OpenRouterModel {
    fn name(&self) -> &str { &self.model_name }
    fn system(&self) -> &str { "openrouter" }
    fn profile(&self) -> &ModelProfile { &self.profile }

    async fn request(
        &self, messages: &[ModelRequest], settings: &ModelSettings, params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let body = self.build_request(messages, settings, params, false);
        let response = self.send_request(&body, settings.timeout.unwrap_or(self.default_timeout)).await?;
        let resp: ChatCompletionResponse = response.json().await
            .map_err(|e| ModelError::invalid_response(e.to_string()))?;
        self.parse_response(resp)
    }

    async fn request_stream(
        &self, messages: &[ModelRequest], settings: &ModelSettings, params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        let body = self.build_request(messages, settings, params, true);
        let response = self.send_request(&body, settings.timeout.unwrap_or(self.default_timeout)).await?;
        Ok(Box::pin(OpenAIStreamParser::new(response.bytes_stream())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openrouter() {
        let model = OpenRouterModel::new("anthropic/claude-3-opus", "key")
            .with_http_referer("https://myapp.com").with_app_title("My App")
            .with_provider_preferences(ProviderPreferences::new().with_order(vec!["anthropic".into()]))
            .with_transforms(vec!["middle-out".into()]);
        assert_eq!(model.name(), "anthropic/claude-3-opus");
        assert_eq!(model.system(), "openrouter");
        let mut req = ModelRequest::new(); req.add_user_prompt("Hi");
        let body = model.build_request(&[req], &ModelSettings::new(), &ModelRequestParameters::new(), false);
        assert!(body.get("provider").is_some() && body.get("transforms").is_some());
    }
}
