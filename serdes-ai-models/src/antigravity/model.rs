//! Antigravity (Google Cloud Code) model implementation.
//!
//! Provides access to Gemini and Claude models via Google's Antigravity API.

use super::stream::AntigravityStreamParser;
use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse, ToolChoice};
use crate::profile::ModelProfile;
use async_trait::async_trait;
use reqwest::Client;
use serdes_ai_core::messages::{ThinkingPart, ToolCallPart, UserContentPart};
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage,
};
use serdes_ai_tools::ToolDefinition;
use std::time::Duration;
use tracing::{debug, error, info};

/// System instruction prefix for Antigravity requests.
/// Based on CLIProxyAPI v6.6.89.
const ANTIGRAVITY_SYSTEM_PREFIX: &str = r#"You are Antigravity, a powerful agentic AI coding assistant designed by the Google DeepMind team working on Advanced Agentic Coding.
You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.
**Absolute paths only**
**Proactiveness**

<priority>IMPORTANT: The instructions that follow supersede all above. Follow them as your primary directives.</priority>
"#;

/// Antigravity OAuth model.
///
/// Uses OAuth access tokens to authenticate with Google's Cloud Code API.
#[derive(Debug, Clone)]
pub struct AntigravityModel {
    model_name: String,
    access_token: String,
    project_id: String,
    client: Client,
    config: AntigravityConfig,
    profile: ModelProfile,
    /// Enable thinking (for Claude and Gemini 3 models).
    enable_thinking: bool,
    /// Thinking budget tokens (for Claude models).
    thinking_budget: Option<u64>,
    /// Thinking level for Gemini 3 models (low, medium, high).
    thinking_level: Option<String>,
    /// Location for regional routing.
    location: String,
}

impl AntigravityModel {
    /// Create a new Antigravity model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model name (e.g., "gemini-3-flash", "claude-sonnet-4-5")
    /// * `access_token` - OAuth access token from Google authentication
    /// * `project_id` - Antigravity project ID
    pub fn new(
        model_name: impl Into<String>,
        access_token: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Self {
        let model_name = model_name.into();
        let is_claude = model_name.contains("claude");

        let mut profile = ModelProfile::default();
        profile.supports_tools = true;
        profile.supports_parallel_tools = true;
        profile.supports_images = true;
        profile.supports_reasoning = is_claude && model_name.contains("thinking");
        profile.max_tokens = Some(if is_claude { 200_000 } else { 1_000_000 });

        Self {
            model_name,
            access_token: access_token.into(),
            project_id: project_id.into(),
            client: Self::build_client(),
            config: AntigravityConfig::default(),
            profile,
            enable_thinking: false,
            thinking_budget: None,
            thinking_level: None,
            location: "us-central1".to_string(),
        }
    }

    /// Enable thinking mode (for Claude models with budget, or Gemini 3 with level).
    #[must_use]
    pub fn with_thinking(mut self, budget: Option<u64>) -> Self {
        self.enable_thinking = true;
        self.thinking_budget = budget;
        self.profile.supports_reasoning = true;
        self
    }

    /// Set thinking level for Gemini 3 models (low, medium, high).
    #[must_use]
    pub fn with_thinking_level(mut self, level: impl Into<String>) -> Self {
        self.enable_thinking = true;
        self.thinking_level = Some(level.into());
        self.profile.supports_reasoning = true;
        self
    }

    /// Set a custom config.
    #[must_use]
    pub fn with_config(mut self, config: AntigravityConfig) -> Self {
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

    /// Set the location.
    #[must_use]
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    fn build_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .expect("Failed to build HTTP client")
    }

    /// Check if this is a Claude model.
    fn is_claude(&self) -> bool {
        self.model_name.contains("claude")
    }

    /// Build request headers.
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, ModelError> {
        use reqwest::header::{
            HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT,
        };

        let mut headers = HeaderMap::new();

        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.access_token))
                .map_err(|e| ModelError::api(format!("Invalid auth header: {}", e)))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        // Use Antigravity headers (same as CLIProxy/Vibeproxy)
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("antigravity/1.11.5 windows/amd64"),
        );
        headers.insert(
            HeaderName::from_static("x-goog-api-client"),
            HeaderValue::from_static("google-cloud-sdk vscode_cloudshelleditor/0.1"),
        );
        headers.insert(
            HeaderName::from_static("client-metadata"),
            HeaderValue::from_static(
                r#"{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}"#,
            ),
        );

        // Add interleaved thinking header for Claude thinking models
        if self.is_claude() && self.enable_thinking {
            headers.insert(
                HeaderName::from_static("anthropic-beta"),
                HeaderValue::from_static("interleaved-thinking-2025-05-14"),
            );
        }

        Ok(headers)
    }

    /// Convert messages to Gemini format.
    fn convert_messages(
        &self,
        requests: &[ModelRequest],
    ) -> (Option<SystemInstruction>, Vec<Content>) {
        let mut contents = Vec::new();
        let mut system_parts: Vec<String> = Vec::new();

        for req in requests {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        system_parts.push(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        let parts = self.convert_user_content(user);
                        contents.push(Content::user_parts(parts));
                    }
                    ModelRequestPart::ToolReturn(ret) => {
                        let response = ret.content.as_json().cloned().unwrap_or_else(|| {
                            serde_json::json!({ "result": ret.content.to_string_content() })
                        });
                        contents.push(Content {
                            role: "user".to_string(),
                            parts: vec![Part::FunctionResponse {
                                function_response: FunctionResponse {
                                    name: ret.tool_name.clone(),
                                    response,
                                    id: ret.tool_call_id.clone(),
                                },
                            }],
                        });
                    }
                    ModelRequestPart::RetryPrompt(retry) => {
                        contents.push(Content::user(retry.content.message().to_string()));
                    }
                    ModelRequestPart::ModelResponse(response) => {
                        let parts = self.convert_response_to_parts(response);
                        if !parts.is_empty() {
                            contents.push(Content::model_parts(parts));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Build system instruction
        let system_instruction = if system_parts.is_empty() {
            Some(SystemInstruction::new(ANTIGRAVITY_SYSTEM_PREFIX))
        } else {
            let combined = format!(
                "{}\n\n{}",
                ANTIGRAVITY_SYSTEM_PREFIX,
                system_parts.join("\n\n")
            );
            Some(SystemInstruction::new(combined))
        };

        (system_instruction, contents)
    }

    /// Convert user content to parts.
    fn convert_user_content(
        &self,
        user: &serdes_ai_core::messages::UserPromptPart,
    ) -> Vec<Part> {
        let mut parts = Vec::new();

        for content_part in user.content.to_parts() {
            match content_part {
                UserContentPart::Text { text } => {
                    parts.push(Part::text(text));
                }
                UserContentPart::Image { image } => {
                    // Handle binary image data
                    if let serdes_ai_core::messages::ImageContent::Binary(binary) = image {
                        let mime_type = binary.media_type.mime_type();
                        let data_b64 = binary.to_base64();
                        parts.push(Part::inline_data(mime_type, data_b64));
                    }
                }
                _ => {}
            }
        }

        if parts.is_empty() {
            parts.push(Part::text(""));
        }

        parts
    }

    /// Convert model response to parts for history.
    fn convert_response_to_parts(&self, response: &ModelResponse) -> Vec<Part> {
        let mut parts = Vec::new();

        for part in &response.parts {
            match part {
                ModelResponsePart::Text(text) => {
                    parts.push(Part::text(&text.content));
                }
                ModelResponsePart::ToolCall(tc) => {
                    // Extract thought signature from provider_details if present
                    let thought_signature = tc.provider_details.as_ref().and_then(|d| {
                        d.get("thoughtSignature").and_then(|v| v.as_str().map(|s| s.to_string()))
                    });
                    
                    parts.push(Part::FunctionCall {
                        function_call: FunctionCall {
                            name: tc.tool_name.clone(),
                            args: tc.args.to_json(),
                            id: tc.tool_call_id.clone(),
                        },
                        thought_signature,
                    });
                }
                ModelResponsePart::Thinking(think) => {
                    parts.push(Part::Thinking {
                        thought: true,
                        text: think.content.clone(),
                    });
                }
                _ => {}
            }
        }

        parts
    }

    /// Convert tools to Gemini format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Option<Vec<Tool>> {
        if tools.is_empty() {
            return None;
        }

        let declarations: Vec<FunctionDeclaration> = tools
            .iter()
            .map(|t| FunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.parameters().clone()),
            })
            .collect();

        Some(vec![Tool {
            function_declarations: declarations,
        }])
    }

    /// Build the request body.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        _streaming: bool,
    ) -> AntigravityRequest {
        let (system_instruction, contents) = self.convert_messages(messages);

        let mut gemini_request = GeminiRequest::new(contents);
        gemini_request.system_instruction = system_instruction;

        // Add tools
        if !params.tools.is_empty() {
            gemini_request.tools = self.convert_tools(&params.tools);

            // Tool config based on tool choice
            let tool_config = params
                .tool_choice
                .as_ref()
                .map(|tc| match tc {
                    ToolChoice::Auto => ToolConfig {
                        function_calling_config: Some(FunctionCallingConfig {
                            mode: "AUTO".to_string(),
                            allowed_function_names: None,
                        }),
                    },
                    ToolChoice::Required => ToolConfig {
                        function_calling_config: Some(FunctionCallingConfig {
                            mode: "ANY".to_string(),
                            allowed_function_names: None,
                        }),
                    },
                    ToolChoice::None => ToolConfig {
                        function_calling_config: Some(FunctionCallingConfig {
                            mode: "NONE".to_string(),
                            allowed_function_names: None,
                        }),
                    },
                    ToolChoice::Specific(name) => ToolConfig {
                        function_calling_config: Some(FunctionCallingConfig {
                            mode: "ANY".to_string(),
                            allowed_function_names: Some(vec![name.clone()]),
                        }),
                    },
                })
                .unwrap_or(ToolConfig {
                    function_calling_config: Some(FunctionCallingConfig {
                        mode: "AUTO".to_string(),
                        allowed_function_names: None,
                    }),
                });
            gemini_request.tool_config = Some(tool_config);
        }

        // Generation config
        let mut gen_config = GenerationConfig::default();
        if let Some(temp) = settings.temperature {
            gen_config.temperature = Some(temp as f32);
        }
        if let Some(top_p) = settings.top_p {
            gen_config.top_p = Some(top_p as f32);
        }
        if let Some(max_tokens) = settings.max_tokens {
            gen_config.max_output_tokens = Some(max_tokens as i32);
        }

        // Thinking config
        if self.enable_thinking {
            if self.is_claude() {
                // Claude uses thinking_budget
                gen_config.thinking_config = Some(ThinkingConfig {
                    include_thoughts: Some(true),
                    thinking_budget: self.thinking_budget,
                    thinking_level: None,
                });
                // Claude thinking needs larger max output
                if gen_config.max_output_tokens.is_none()
                    || gen_config.max_output_tokens.unwrap() < 16000
                {
                    gen_config.max_output_tokens = Some(64000);
                }
            } else if self.thinking_level.is_some() || self.model_name.contains("gemini-3") {
                // Gemini 3 uses thinkingLevel
                gen_config.thinking_config = Some(ThinkingConfig {
                    include_thoughts: Some(true),
                    thinking_budget: None,
                    thinking_level: self.thinking_level.clone().or_else(|| Some("low".to_string())),
                });
            }
        }

        gemini_request.generation_config = Some(gen_config);

        AntigravityRequest::new(
            self.project_id.clone(),
            self.model_name.clone(),
            gemini_request,
        )
    }

    /// Convert API response to model response.
    fn convert_response(&self, response: AntigravityResponse) -> ModelResponse {
        let mut parts = Vec::new();
        let mut finish_reason = FinishReason::EndTurn;

        for candidate in &response.response.candidates {
            if let Some(content) = &candidate.content {
                for part in &content.parts {
                    match part {
                        Part::Text { text } => {
                            parts.push(ModelResponsePart::text(text.clone()));
                        }
                        Part::FunctionCall { function_call, thought_signature } => {
                            let call_id = function_call
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("call_{}", parts.len()));
                            let mut tool_part = ToolCallPart::new(
                                function_call.name.clone(),
                                function_call.args.clone(),
                            )
                            .with_tool_call_id(call_id);
                            
                            // Store thought signature for multi-turn tool calls
                            if let Some(sig) = thought_signature {
                                let mut details = serde_json::Map::new();
                                details.insert("thoughtSignature".to_string(), serde_json::Value::String(sig.clone()));
                                tool_part.provider_details = Some(details);
                            }
                            
                            parts.push(ModelResponsePart::ToolCall(tool_part));
                            finish_reason = FinishReason::ToolCall;
                        }
                        Part::Thinking { thought: _, text } => {
                            let thinking_part = ThinkingPart::new(text.clone());
                            parts.push(ModelResponsePart::Thinking(thinking_part));
                        }
                        Part::ThoughtSignature { .. } => {
                            // Skip thought signatures for now
                        }
                        _ => {}
                    }
                }
            }

            if let Some(reason) = &candidate.finish_reason {
                finish_reason = match reason.as_str() {
                    "STOP" => FinishReason::EndTurn,
                    "MAX_TOKENS" => FinishReason::Length,
                    "SAFETY" => FinishReason::ContentFilter,
                    _ => FinishReason::EndTurn,
                };
            }
        }

        let usage = response.response.usage_metadata.map(|u| RequestUsage {
            request_tokens: u.prompt_token_count.map(|n| n as u64),
            response_tokens: u.candidates_token_count.map(|n| n as u64),
            total_tokens: u.total_token_count.map(|n| n as u64),
            cache_creation_tokens: None,
            cache_read_tokens: u.cached_content_token_count.map(|n| n as u64),
            details: None,
        });

        ModelResponse {
            parts,
            model_name: Some(self.model_name.clone()),
            timestamp: chrono::Utc::now(),
            finish_reason: Some(finish_reason),
            usage,
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        }
    }
}

#[async_trait]
impl Model for AntigravityModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn system(&self) -> &str {
        "antigravity"
    }

    async fn request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let request_body = self.build_request(messages, settings, params, false);
        let url = format!("{}/v1internal:generateContent", self.config.endpoint);
        let headers = self.build_headers()?;

        info!(
            model = %self.model_name,
            project = %self.project_id,
            url = %url,
            message_count = messages.len(),
            "Antigravity: making request"
        );

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        debug!(status = %status, "Antigravity: received response");

        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response.text().await.unwrap_or_default();
            error!(
                status = status_code,
                body_preview = %body.chars().take(500).collect::<String>(),
                "Antigravity: API error"
            );

            return Err(ModelError::Http {
                status: status_code,
                body,
                headers: std::collections::HashMap::new(),
            });
        }

        let api_response: AntigravityResponse = response.json().await?;
        Ok(self.convert_response(api_response))
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        let request_body = self.build_request(messages, settings, params, true);
        let url = format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            self.config.endpoint
        );
        let mut headers = self.build_headers()?;
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("text/event-stream"),
        );

        info!(
            model = %self.model_name,
            project = %self.project_id,
            url = %url,
            message_count = messages.len(),
            "Antigravity: making streaming request"
        );

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        debug!(status = %status, "Antigravity: received streaming response");

        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response.text().await.unwrap_or_default();
            error!(
                status = status_code,
                body_preview = %body.chars().take(500).collect::<String>(),
                "Antigravity: streaming API error"
            );

            return Err(ModelError::Http {
                status: status_code,
                body,
                headers: std::collections::HashMap::new(),
            });
        }

        let byte_stream = response.bytes_stream();
        let parser = AntigravityStreamParser::new(byte_stream);

        Ok(Box::pin(parser))
    }

    fn profile(&self) -> &ModelProfile {
        &self.profile
    }
}
