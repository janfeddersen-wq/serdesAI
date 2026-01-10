//! AWS Bedrock model implementation.
//!
//! [Amazon Bedrock](https://aws.amazon.com/bedrock/) provides access to foundation
//! models from Amazon, Anthropic, Meta, Mistral, and others.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_models::bedrock::BedrockModel;
//!
//! // Uses AWS credentials from environment/config
//! let model = BedrockModel::new("anthropic.claude-3-sonnet-20240229-v1:0")?;
//!
//! // Custom region
//! let model = BedrockModel::new("meta.llama3-70b-instruct-v1:0")?
//!     .with_region("eu-west-1");
//! ```
//!
//! ## Available Models
//!
//! ### Anthropic Claude
//! - `anthropic.claude-3-5-sonnet-20241022-v2:0` - Claude 3.5 Sonnet v2
//! - `anthropic.claude-3-opus-20240229-v1:0` - Claude 3 Opus
//! - `anthropic.claude-3-haiku-20240307-v1:0` - Claude 3 Haiku
//!
//! ### Meta Llama
//! - `meta.llama3-1-405b-instruct-v1:0` - Llama 3.1 405B
//! - `meta.llama3-1-70b-instruct-v1:0` - Llama 3.1 70B
//! - `meta.llama3-1-8b-instruct-v1:0` - Llama 3.1 8B
//!
//! ### Mistral
//! - `mistral.mistral-large-2407-v1:0` - Mistral Large
//! - `mistral.mixtral-8x7b-instruct-v0:1` - Mixtral 8x7B
//!
//! ### Amazon Titan
//! - `amazon.titan-text-premier-v1:0` - Titan Text Premier
//! - `amazon.titan-text-express-v1` - Titan Text Express

pub mod types;

use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::ModelProfile;
use serdes_ai_core::{
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage, TextPart, ThinkingPart, ToolCallPart, UserContent, UserContentPart,
};

/// AWS Bedrock model client.
#[derive(Debug, Clone)]
pub struct BedrockModel {
    /// Model ID.
    model_id: String,
    /// HTTP client.
    client: Client,
    /// AWS region.
    region: String,
    /// AWS credentials.
    #[allow(dead_code)] // Used for future AWS SigV4 signing
    credentials: AwsCredentials,
    /// Model profile.
    profile: ModelProfile,
    /// Default timeout.
    default_timeout: Duration,
}

/// AWS credentials.
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    /// Access key ID.
    pub access_key_id: String,
    /// Secret access key.
    pub secret_access_key: String,
    /// Session token (optional).
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// Create new credentials.
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
        }
    }

    /// With session token.
    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    /// Load from environment variables.
    pub fn from_env() -> Result<Self, ModelError> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| ModelError::configuration("AWS_ACCESS_KEY_ID not set"))?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| ModelError::configuration("AWS_SECRET_ACCESS_KEY not set"))?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        Ok(Self {
            access_key_id: access_key,
            secret_access_key: secret_key,
            session_token,
        })
    }
}

impl BedrockModel {
    /// Default AWS region.
    pub const DEFAULT_REGION: &'static str = "us-east-1";

    /// Create a new Bedrock model.
    pub fn new(model_id: impl Into<String>) -> Result<Self, ModelError> {
        let credentials = AwsCredentials::from_env()?;
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| Self::DEFAULT_REGION.to_string());

        Ok(Self {
            model_id: model_id.into(),
            client: Client::new(),
            region,
            credentials,
            profile: Self::default_profile(),
            default_timeout: Duration::from_secs(120),
        })
    }

    /// Create with explicit credentials.
    pub fn with_credentials(model_id: impl Into<String>, credentials: AwsCredentials) -> Self {
        Self {
            model_id: model_id.into(),
            client: Client::new(),
            region: Self::DEFAULT_REGION.to_string(),
            credentials,
            profile: Self::default_profile(),
            default_timeout: Duration::from_secs(120),
        }
    }

    /// Set the region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
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

    /// Default profile for Bedrock models.
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

    /// Get the endpoint URL.
    fn endpoint(&self) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse",
            self.region, self.model_id
        )
    }

    /// Detect the model family.
    pub fn model_family(&self) -> ModelFamily {
        if self.model_id.starts_with("anthropic.") {
            ModelFamily::Anthropic
        } else if self.model_id.starts_with("meta.") {
            ModelFamily::Meta
        } else if self.model_id.starts_with("mistral.") {
            ModelFamily::Mistral
        } else if self.model_id.starts_with("amazon.") {
            ModelFamily::Amazon
        } else if self.model_id.starts_with("cohere.") {
            ModelFamily::Cohere
        } else if self.model_id.starts_with("ai21.") {
            ModelFamily::AI21
        } else {
            ModelFamily::Unknown
        }
    }

    /// Build the converse request.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> types::ConverseRequest {
        let api_messages = self.convert_messages(messages);
        let system = self.extract_system_prompt(messages);
        let tool_config = self.build_tool_config(params);

        let inference_config = types::InferenceConfig {
            max_tokens: settings.max_tokens,
            temperature: settings.temperature.map(|t| t as f32),
            top_p: settings.top_p.map(|t| t as f32),
            stop_sequences: settings.stop.clone(),
        };

        types::ConverseRequest {
            model_id: self.model_id.clone(),
            messages: api_messages,
            system,
            inference_config: Some(inference_config),
            tool_config,
        }
    }

    /// Extract system prompt.
    fn extract_system_prompt(
        &self,
        messages: &[ModelRequest],
    ) -> Option<Vec<types::SystemContent>> {
        let system_parts: Vec<_> = messages
            .iter()
            .flat_map(|req| &req.parts)
            .filter_map(|p| {
                if let ModelRequestPart::SystemPrompt(sp) = p {
                    Some(types::SystemContent::Text {
                        text: sp.content.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        if system_parts.is_empty() {
            None
        } else {
            Some(system_parts)
        }
    }

    /// Convert messages.
    fn convert_messages(&self, messages: &[ModelRequest]) -> Vec<types::Message> {
        let mut result = Vec::new();

        for request in messages {
            for part in &request.parts {
                match part {
                    ModelRequestPart::UserPrompt(up) => {
                        let content = self.convert_user_content(&up.content);
                        result.push(types::Message {
                            role: types::Role::User,
                            content,
                        });
                    }
                    ModelRequestPart::ToolReturn(tr) => {
                        result.push(types::Message {
                            role: types::Role::User,
                            content: vec![types::Content::ToolResult {
                                tool_use_id: tr.tool_call_id.clone().unwrap_or_default(),
                                content: vec![types::ToolResultContent::Text {
                                    text: tr.content.to_string_content(),
                                }],
                            }],
                        });
                    }
                    ModelRequestPart::RetryPrompt(rp) => {
                        result.push(types::Message {
                            role: types::Role::User,
                            content: vec![types::Content::Text {
                                text: rp.content.message().to_string(),
                            }],
                        });
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        result.push(types::Message {
                            role: types::Role::User,
                            content: vec![types::Content::ToolResult {
                                tool_use_id: builtin.tool_call_id.clone(),
                                content: vec![types::ToolResultContent::Text { text: content_str }],
                            }],
                        });
                    }
                    // System prompts are handled separately
                    ModelRequestPart::SystemPrompt(_) => {}
                    ModelRequestPart::ModelResponse(response) => {
                        // Add assistant response for proper alternation
                        let mut content = Vec::new();
                        for resp_part in &response.parts {
                            match resp_part {
                                serdes_ai_core::ModelResponsePart::Text(t) => {
                                    content.push(types::Content::Text {
                                        text: t.content.clone(),
                                    });
                                }
                                serdes_ai_core::ModelResponsePart::ToolCall(tc) => {
                                    content.push(types::Content::ToolUse {
                                        tool_use_id: tc.tool_call_id.clone().unwrap_or_default(),
                                        name: tc.tool_name.clone(),
                                        input: tc.args.to_json(),
                                    });
                                }
                                _ => {}
                            }
                        }
                        if !content.is_empty() {
                            result.push(types::Message {
                                role: types::Role::Assistant,
                                content,
                            });
                        }
                    }
                }
            }
        }

        result
    }

    /// Convert user content.
    fn convert_user_content(&self, content: &UserContent) -> Vec<types::Content> {
        match content {
            UserContent::Text(t) => {
                vec![types::Content::Text { text: t.clone() }]
            }
            UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    UserContentPart::Text { text } => {
                        Some(types::Content::Text { text: text.clone() })
                    }
                    UserContentPart::Image { image } => match image {
                        serdes_ai_core::messages::ImageContent::Binary(binary) => {
                            use base64::Engine;
                            let encoded =
                                base64::engine::general_purpose::STANDARD.encode(&binary.data);
                            Some(types::Content::Image {
                                image: types::ImageBlock {
                                    format: binary.media_type.extension().to_string(),
                                    source: types::ImageSource::Bytes { bytes: encoded },
                                },
                            })
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .collect(),
        }
    }

    /// Build tool config.
    fn build_tool_config(&self, params: &ModelRequestParameters) -> Option<types::ToolConfig> {
        if params.tools.is_empty() {
            return None;
        }

        let tools: Vec<types::Tool> = params
            .tools
            .iter()
            .map(|t| types::Tool {
                tool_spec: types::ToolSpec {
                    name: t.name.clone(),
                    description: Some(t.description.clone()),
                    input_schema: types::ToolInputSchema {
                        json: serde_json::to_value(&t.parameters_json_schema).unwrap_or_default(),
                    },
                },
            })
            .collect();

        Some(types::ToolConfig {
            tools,
            tool_choice: None,
        })
    }

    /// Parse response.
    fn parse_response(
        &self,
        response: types::ConverseResponse,
    ) -> Result<ModelResponse, ModelError> {
        let mut parts = Vec::new();

        if let Some(output) = response.output {
            if let Some(message) = output.message {
                for content in message.content {
                    match content {
                        types::Content::Text { text } => {
                            parts.push(ModelResponsePart::Text(TextPart::new(text)));
                        }
                        types::Content::ToolUse {
                            tool_use_id,
                            name,
                            input,
                        } => {
                            parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                                tool_name: name,
                                args: serdes_ai_core::messages::ToolCallArgs::Json(input),
                                tool_call_id: Some(tool_use_id),
                                id: None,
                                provider_details: None,
                            }));
                        }
                        types::Content::ReasoningContent {
                            reasoning_text,
                            redacted_content,
                        } => {
                            // Handle reasoning/thinking content from Claude via Bedrock
                            if let Some(redacted) = redacted_content {
                                // Redacted thinking - preserve the signature
                                parts.push(ModelResponsePart::Thinking(ThinkingPart::redacted(
                                    redacted, "bedrock",
                                )));
                            } else if let Some(reasoning) = reasoning_text {
                                // Regular thinking content
                                let mut thinking = ThinkingPart::new(&reasoning.text);
                                if let Some(sig) = reasoning.signature {
                                    thinking = thinking.with_signature(sig);
                                }
                                parts.push(ModelResponsePart::Thinking(thinking));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let finish_reason = match response.stop_reason.as_deref() {
            Some("end_turn") | Some("stop") => Some(FinishReason::EndTurn),
            Some("max_tokens") => Some(FinishReason::Length),
            Some("tool_use") => Some(FinishReason::ToolCall),
            _ => None,
        };

        let usage = response.usage.map(|u| RequestUsage {
            request_tokens: Some(u.input_tokens as u64),
            response_tokens: Some(u.output_tokens as u64),
            total_tokens: Some((u.input_tokens + u.output_tokens) as u64),
            ..Default::default()
        });

        Ok(ModelResponse {
            parts,
            finish_reason,
            usage,
            model_name: Some(self.model_id.clone()),
            timestamp: serdes_ai_core::identifier::now_utc(),
            vendor_id: None,
            vendor_details: None,
            kind: "response".to_string(),
        })
    }
}

/// Model family for Bedrock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Anthropic Claude.
    Anthropic,
    /// Meta Llama.
    Meta,
    /// Mistral.
    Mistral,
    /// Amazon Titan.
    Amazon,
    /// Cohere.
    Cohere,
    /// AI21.
    AI21,
    /// Unknown.
    Unknown,
}

#[async_trait]
impl Model for BedrockModel {
    fn name(&self) -> &str {
        &self.model_id
    }

    fn system(&self) -> &str {
        "bedrock"
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

        // Note: In production, you would use AWS SigV4 signing.
        // This is a simplified implementation.
        let response = self
            .client
            .post(self.endpoint())
            .header("Content-Type", "application/json")
            .timeout(timeout)
            // AWS SigV4 headers would go here
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(ModelError::http(status, text));
        }

        let converse_response: types::ConverseResponse = response
            .json()
            .await
            .map_err(|e| ModelError::invalid_response(e.to_string()))?;

        self.parse_response(converse_response)
    }

    async fn request_stream(
        &self,
        _messages: &[ModelRequest],
        _settings: &ModelSettings,
        _params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        Err(ModelError::not_supported("Streaming for Bedrock"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_family_detection() {
        let model = BedrockModel::with_credentials(
            "anthropic.claude-3-sonnet",
            AwsCredentials::new("key", "secret"),
        );
        assert_eq!(model.model_family(), ModelFamily::Anthropic);

        let model =
            BedrockModel::with_credentials("meta.llama3-70b", AwsCredentials::new("key", "secret"));
        assert_eq!(model.model_family(), ModelFamily::Meta);
    }

    #[test]
    fn test_credentials() {
        let creds = AwsCredentials::new("access", "secret").with_session_token("token");
        assert_eq!(creds.access_key_id, "access");
        assert_eq!(creds.session_token, Some("token".to_string()));
    }
}
