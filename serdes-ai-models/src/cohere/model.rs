//! Cohere model implementation.
//!
//! Cohere provides powerful language models including Command-R series
//! with native RAG and tool-use capabilities.

use super::types::*;
use crate::error::ModelError;
use crate::model::{Model, ModelRequestParameters, StreamedResponse};
use crate::profile::ModelProfile;
use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use reqwest::Client;
use serdes_ai_core::{
    messages::{
        ModelResponseStreamEvent, TextPart, ToolCallArgs, ToolCallPart, UserContent,
        UserContentPart,
    },
    FinishReason, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart, ModelSettings,
    RequestUsage,
};
use serdes_ai_tools::ToolDefinition;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

const COHERE_BASE_URL: &str = "https://api.cohere.ai/v2";

/// Cohere model client.
#[derive(Debug, Clone)]
pub struct CohereModel {
    model_name: String,
    client: Client,
    api_key: String,
    base_url: String,
    profile: ModelProfile,
    default_timeout: Duration,
}

impl CohereModel {
    /// Create a new Cohere model.
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model_name = model_name.into();
        let profile = Self::profile_for_model(&model_name);
        Self {
            model_name,
            client: Client::new(),
            api_key: api_key.into(),
            base_url: COHERE_BASE_URL.into(),
            profile,
            default_timeout: Duration::from_secs(120),
        }
    }

    /// Create from environment variable `CO_API_KEY`.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("CO_API_KEY")
            .map_err(|_| ModelError::configuration("CO_API_KEY not set"))?;
        Ok(Self::new(model_name, api_key))
    }

    /// Create a Command-R Plus model.
    pub fn command_r_plus(api_key: impl Into<String>) -> Self {
        Self::new("command-r-plus", api_key)
    }

    /// Create a Command-R model.
    pub fn command_r(api_key: impl Into<String>) -> Self {
        Self::new("command-r", api_key)
    }

    /// Create a Command-R Plus model from environment.
    pub fn command_r_plus_from_env() -> Result<Self, ModelError> {
        Self::from_env("command-r-plus")
    }

    /// Determine profile based on model name.
    fn profile_for_model(model: &str) -> ModelProfile {
        let mut profile = ModelProfile::new()
            .with_tools(true)
            .with_parallel_tools(true)
            .with_native_structured_output(false);

        // Command-R models have good context windows
        if model.contains("command-r") {
            profile.context_window = Some(128_000);
            profile.max_tokens = Some(4096);
        }

        profile
    }

    /// Convert internal messages to Cohere format.
    fn convert_messages(
        &self,
        requests: &[ModelRequest],
    ) -> (String, Option<Vec<ChatMessage>>, Option<String>) {
        let mut history = Vec::new();
        let mut system_prompt = None;
        let mut current_message = String::new();

        for req in requests {
            for part in &req.parts {
                match part {
                    ModelRequestPart::SystemPrompt(sys) => {
                        system_prompt = Some(sys.content.clone());
                    }
                    ModelRequestPart::UserPrompt(user) => {
                        // If we already have a current message, push it to history
                        if !current_message.is_empty() {
                            history.push(ChatMessage::user(&current_message));
                            current_message.clear();
                        }
                        current_message = match &user.content {
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
                    }
                    ModelRequestPart::ToolReturn(ret) => {
                        // Tool returns become part of the conversation
                        let content = ret.content.to_string_content();
                        history.push(ChatMessage {
                            role: Role::Tool,
                            message: content,
                            tool_calls: None,
                        });
                    }
                    ModelRequestPart::RetryPrompt(retry) => {
                        current_message = retry.content.message().to_string();
                    }
                    ModelRequestPart::BuiltinToolReturn(builtin) => {
                        // Convert builtin tool return to tool result
                        let content_str = serde_json::to_string(&builtin.content)
                            .unwrap_or_else(|_| builtin.content_type().to_string());
                        history.push(ChatMessage {
                            role: Role::Tool,
                            message: content_str,
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
                        if !text_content.is_empty() {
                            history.push(ChatMessage {
                                role: Role::Chatbot,
                                message: text_content,
                                tool_calls: None,
                            });
                        }
                    }
                }
            }
        }

        let history = if history.is_empty() {
            None
        } else {
            Some(history)
        };
        (current_message, history, system_prompt)
    }

    /// Convert tool definitions to Cohere format.
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<Tool> {
        tools
            .iter()
            .map(|t| Tool {
                name: t.name.clone(),
                description: t.description.clone(),
                parameter_definitions: Some(
                    serde_json::to_value(&t.parameters_json_schema).unwrap_or_default(),
                ),
            })
            .collect()
    }

    /// Build the chat request.
    fn build_request(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
        stream: bool,
    ) -> ChatRequest {
        let (message, chat_history, preamble) = self.convert_messages(messages);

        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools(&params.tools))
        };

        ChatRequest {
            model: self.model_name.clone(),
            message,
            chat_history,
            preamble,
            temperature: settings.temperature.map(|t| t as f32),
            max_tokens: settings.max_tokens.map(|t| t as u32),
            p: settings.top_p.map(|t| t as f32),
            k: None,
            stop_sequences: settings.stop.clone(),
            stream: if stream { Some(true) } else { None },
            tools,
            tool_results: None,
        }
    }

    /// Parse a non-streaming response.
    fn parse_response(&self, resp: ChatResponse) -> Result<ModelResponse, ModelError> {
        let mut parts = Vec::new();

        // Add text content
        if !resp.text.is_empty() {
            parts.push(ModelResponsePart::Text(TextPart::new(&resp.text)));
        }

        // Add tool calls
        if let Some(tool_calls) = resp.tool_calls {
            for tc in tool_calls {
                parts.push(ModelResponsePart::ToolCall(
                    ToolCallPart::new(&tc.name, ToolCallArgs::Json(tc.parameters))
                        .with_tool_call_id(tc.id),
                ));
            }
        }

        let finish_reason = resp.finish_reason.map(|r| match r.as_str() {
            "COMPLETE" | "END_TURN" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "TOOL_CALL" => FinishReason::ToolCall,
            _ => FinishReason::Stop,
        });

        let usage = resp.meta.and_then(|m| m.tokens).map(|t| RequestUsage {
            request_tokens: t.input_tokens.map(u64::from),
            response_tokens: t.output_tokens.map(u64::from),
            total_tokens: t
                .input_tokens
                .zip(t.output_tokens)
                .map(|(a, b)| u64::from(a) + u64::from(b)),
            cache_creation_tokens: None,
            cache_read_tokens: None,
            details: None,
        });

        Ok(ModelResponse {
            parts,
            model_name: Some(self.model_name.clone()),
            timestamp: chrono::Utc::now(),
            finish_reason,
            usage,
            vendor_id: resp.generation_id,
            vendor_details: None,
            kind: "response".into(),
        })
    }

    fn handle_error(&self, status: u16, body: &str) -> ModelError {
        serde_json::from_str::<CohereError>(body)
            .map(|e| match status {
                401 => ModelError::auth(e.message),
                429 => ModelError::rate_limited(None),
                404 => ModelError::NotFound(e.message),
                _ => ModelError::Api {
                    message: e.message,
                    code: None,
                },
            })
            .unwrap_or_else(|_| ModelError::http(status, body))
    }

    async fn send_request(
        &self,
        body: &ChatRequest,
        timeout: Duration,
    ) -> Result<reqwest::Response, ModelError> {
        let response = self
            .client
            .post(format!("{}/chat", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(timeout)
            .json(body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(self.handle_error(status, &body));
        }
        Ok(response)
    }
}

#[async_trait]
impl Model for CohereModel {
    fn name(&self) -> &str {
        &self.model_name
    }
    fn system(&self) -> &str {
        "cohere"
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
        let body = self.build_request(messages, settings, params, false);
        let response = self
            .send_request(&body, settings.timeout.unwrap_or(self.default_timeout))
            .await?;
        let resp: ChatResponse = response
            .json()
            .await
            .map_err(|e| ModelError::invalid_response(e.to_string()))?;
        self.parse_response(resp)
    }

    async fn request_stream(
        &self,
        messages: &[ModelRequest],
        settings: &ModelSettings,
        params: &ModelRequestParameters,
    ) -> Result<StreamedResponse, ModelError> {
        let body = self.build_request(messages, settings, params, true);
        let response = self
            .send_request(&body, settings.timeout.unwrap_or(self.default_timeout))
            .await?;
        Ok(Box::pin(CohereStreamParser::new(response.bytes_stream())))
    }
}

/// Stream parser for Cohere SSE responses.
pub struct CohereStreamParser<S> {
    inner: S,
    buffer: String,
    started: bool,
}

impl<S> CohereStreamParser<S> {
    /// Create a new Cohere stream parser.
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            buffer: String::new(),
            started: false,
        }
    }
}

impl<S> Stream for CohereStreamParser<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ModelResponseStreamEvent, ModelError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Try to parse a complete event from buffer
            if let Some(event) = self.try_parse_event() {
                return Poll::Ready(Some(Ok(event)));
            }

            // Need more data
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    self.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ModelError::Network(e.to_string()))));
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<S> CohereStreamParser<S> {
    fn try_parse_event(&mut self) -> Option<ModelResponseStreamEvent> {
        // Cohere uses newline-delimited JSON
        while let Some(pos) = self.buffer.find('\n') {
            let line = self.buffer[..pos].trim().to_string();
            self.buffer = self.buffer[pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Ok(event) = serde_json::from_str::<StreamEvent>(&line) {
                match event.event_type.as_str() {
                    "text-generation" => {
                        if let Some(text) = event.text {
                            // Start part if we haven't yet
                            if !self.started {
                                self.started = true;
                                return Some(ModelResponseStreamEvent::part_start(
                                    0,
                                    ModelResponsePart::Text(TextPart::new("")),
                                ));
                            }
                            return Some(ModelResponseStreamEvent::text_delta(0, &text));
                        }
                    }
                    "stream-end" => {
                        return Some(ModelResponseStreamEvent::part_end(0));
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_model_creation() {
        let model = CohereModel::new("command-r-plus", "test-key");
        assert_eq!(model.name(), "command-r-plus");
        assert_eq!(model.system(), "cohere");
    }

    #[test]
    fn test_convenience_constructors() {
        let model = CohereModel::command_r_plus("key");
        assert_eq!(model.name(), "command-r-plus");

        let model = CohereModel::command_r("key");
        assert_eq!(model.name(), "command-r");
    }

    #[test]
    fn test_profile_for_model() {
        let profile = CohereModel::profile_for_model("command-r-plus");
        assert!(profile.supports_tools);
        assert_eq!(profile.context_window, Some(128_000));
    }
}
