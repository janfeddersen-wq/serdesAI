//! Conversions between Open Responses wire types and serdesAI core types.
//!
//! This is the request and response mapping every transport shares: the
//! HTTP request path, the websocket transport, and (from S7) the rig-only
//! protocol helpers all go through these functions.

use super::wire::*;
use crate::error::ModelError;
use crate::model::ToolChoice;
use base64::Engine as _;
use serdes_ai_core::messages::{
    BuiltinToolReturnPart, ImageContent, ModelRequest, ModelRequestPart, ModelResponse,
    ModelResponsePart, TextPart, ThinkingPart, ToolCallArgs, ToolCallPart, ToolReturnContent,
    ToolReturnPart, UserContent, UserContentPart,
};
use serdes_ai_tools::ToolDefinition;

/// Client-side conversion failures.
///
/// The client conversion half only produces invalid-request failures, so
/// this mirrors the protocol crate's `ResponsesError` with that single
/// variant; the code keeps its stable wire spelling for error mapping.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ResponsesError {
    /// The request is malformed or uses an unsupported feature.
    #[error("{0}")]
    InvalidRequest(String),
}

impl ResponsesError {
    /// Stable wire error code for the error.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => codes::INVALID_REQUEST_ERROR,
        }
    }
}

/// Map a serdesAI tool definition onto the wire function tool form.
pub(crate) fn tool_to_wire(tool: &ToolDefinition) -> ResponsesTool {
    ResponsesTool::Function {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters_json_schema.clone(),
        strict: tool.strict,
    }
}

/// Map serdesAI conversation history onto `(instructions, input items)` for a
/// client request.
///
/// The first `skip` requests are treated as already delivered on the session
/// and excluded from the input items, which is what makes websocket
/// continuation turns send only the new material. Instructions are always
/// derived from the full history because the Responses API replaces (rather
/// than appends) instructions on chained turns.
pub(crate) fn history_to_wire(
    messages: &[ModelRequest],
    skip: usize,
) -> Result<(Option<String>, Vec<InputItem>), ResponsesError> {
    let mut instructions: Vec<String> = Vec::new();
    let mut items = Vec::new();

    for (index, request) in messages.iter().enumerate() {
        for part in &request.parts {
            match part {
                ModelRequestPart::SystemPrompt(system) => {
                    instructions.push(system.content.clone());
                }
                ModelRequestPart::UserPrompt(prompt) if index >= skip => {
                    items.push(InputItem::Easy(EasyInputMessage {
                        role: InputRole::User,
                        content: Some(user_content_to_wire(&prompt.content)?),
                    }));
                }
                ModelRequestPart::ToolReturn(tool_return) if index >= skip => {
                    items.push(function_call_output_item(tool_return));
                }
                ModelRequestPart::BuiltinToolReturn(tool_return) if index >= skip => {
                    items.push(builtin_tool_return_item(tool_return));
                }
                ModelRequestPart::ModelResponse(response) if index >= skip => {
                    items.extend(response_parts_to_items(response));
                }
                ModelRequestPart::RetryPrompt(retry) if index >= skip => {
                    tracing::debug!(tool_call_id = ?retry.tool_call_id, "dropping retry prompt from responses input");
                }
                _ => {}
            }
        }
    }

    let instructions = if instructions.is_empty() {
        None
    } else {
        Some(instructions.join("\n\n"))
    };
    Ok((instructions, items))
}

fn user_content_to_wire(content: &UserContent) -> Result<InputMessageContent, ResponsesError> {
    match content {
        UserContent::Text(text) => Ok(InputMessageContent::Text(text.clone())),
        UserContent::Parts(parts) => {
            let mut converted = Vec::new();
            for part in parts {
                match part {
                    UserContentPart::Text { text } => {
                        converted.push(InputContentPart::InputText { text: text.clone() })
                    }
                    UserContentPart::Image { image } => {
                        let url = match image {
                            ImageContent::Url(url) => url.url.clone(),
                            ImageContent::Binary(binary) => format!(
                                "data:{};base64,{}",
                                binary.media_type,
                                base64::engine::general_purpose::STANDARD.encode(&binary.data)
                            ),
                        };
                        converted.push(InputContentPart::InputImage {
                            image_url: InputImageUrl::Url(url),
                            detail: None,
                        });
                    }
                    UserContentPart::Video { .. }
                    | UserContentPart::Document { .. }
                    | UserContentPart::File { .. } => {
                        return Err(ResponsesError::InvalidRequest(
                            "video/document/file input parts are not supported by the responses protocol client"
                                .to_string(),
                        ));
                    }
                    UserContentPart::Audio { .. } => {
                        return Err(ResponsesError::InvalidRequest(
                            "audio input parts are not supported by the responses protocol client"
                                .to_string(),
                        ));
                    }
                }
            }
            Ok(InputMessageContent::Parts(converted))
        }
    }
}

fn function_call_output_item(tool_return: &ToolReturnPart) -> InputItem {
    let output = match &tool_return.content {
        ToolReturnContent::Text { content } => content.clone(),
        ToolReturnContent::Json { content } => content.to_string(),
        ToolReturnContent::Error { error } => format!("tool error: {}", error.message),
        ToolReturnContent::Multiple { .. } | ToolReturnContent::Image { .. } => {
            "[non-text tool output]".to_string()
        }
    };
    InputItem::Typed(TypedInputItem::FunctionCallOutput {
        call_id: tool_return
            .tool_call_id
            .clone()
            .unwrap_or_else(|| tool_return.tool_name.clone()),
        output,
    })
}

fn builtin_tool_return_item(tool_return: &BuiltinToolReturnPart) -> InputItem {
    InputItem::Typed(TypedInputItem::FunctionCallOutput {
        call_id: tool_return.tool_call_id.clone(),
        output: serde_json::to_string(&tool_return.content).unwrap_or_default(),
    })
}

fn response_parts_to_items(response: &ModelResponse) -> Vec<InputItem> {
    response
        .parts
        .iter()
        .filter_map(|part| match part {
            ModelResponsePart::Text(text) => Some(InputItem::Typed(TypedInputItem::Message {
                role: InputRole::Assistant,
                content: Some(InputMessageContent::Parts(vec![
                    InputContentPart::OutputText {
                        text: text.content.clone(),
                    },
                ])),
            })),
            ModelResponsePart::Thinking(thinking) => {
                let encrypted_content = thinking
                    .provider_details
                    .as_ref()
                    .and_then(|details| details.get("encrypted_content"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                Some(InputItem::Typed(TypedInputItem::Reasoning {
                    id: None,
                    summary: vec![SummaryTextItem::new(thinking.content.clone())],
                    encrypted_content,
                }))
            }
            ModelResponsePart::ToolCall(call) => {
                Some(InputItem::Typed(TypedInputItem::FunctionCall {
                    call_id: call
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| call.tool_name.clone()),
                    name: call.tool_name.clone(),
                    arguments: call
                        .args
                        .to_json_string()
                        .unwrap_or_else(|_| "{}".to_string()),
                }))
            }
            ModelResponsePart::File(_) | ModelResponsePart::BuiltinToolCall(_) => None,
        })
        .collect()
}

impl From<ResponsesError> for ModelError {
    fn from(error: ResponsesError) -> Self {
        ModelError::provider(
            "openai",
            error.code(),
            error.to_string(),
            super::events::failure_kind(error.code()),
            None,
        )
    }
}

/// Map completed response output items onto model response parts.
///
/// This is the non-streaming (HTTP) output mapping: reasoning summaries
/// become thinking parts with any encrypted content stashed in provider
/// details for replay on chained turns, message text becomes text parts,
/// refusals surface as content-filter errors, and function calls become
/// tool call parts. Provider-internal items (web search, code interpreter,
/// file search, image generation, MCP calls and their outputs) carry no
/// user-visible part.
pub(crate) fn parts_from_output(
    output: Vec<super::ResponseOutputItem>,
) -> Result<Vec<ModelResponsePart>, ModelError> {
    use super::{MessageContentItem, ReasoningSummaryItem, ResponseOutputItem};

    let mut parts = Vec::new();
    for item in output {
        match item {
            ResponseOutputItem::Reasoning {
                id,
                summary,
                encrypted_content,
                ..
            } => {
                let content = summary
                    .iter()
                    .map(|s| match s {
                        ReasoningSummaryItem::Text { text } => text.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if content.is_empty() && encrypted_content.is_none() {
                    continue;
                }
                let mut thinking = ThinkingPart::new(content)
                    .with_id(&id)
                    .with_provider_name("openai");
                if let Some(encrypted) = encrypted_content {
                    thinking = thinking.with_provider_details(
                        [(
                            "encrypted_content".to_string(),
                            serde_json::Value::String(encrypted),
                        )]
                        .into_iter()
                        .collect(),
                    );
                }
                parts.push(ModelResponsePart::Thinking(thinking));
            }
            ResponseOutputItem::Message { content, .. } => {
                for item in content {
                    match item {
                        MessageContentItem::Text { text, .. } => {
                            if !text.is_empty() {
                                parts.push(ModelResponsePart::Text(TextPart::new(text)));
                            }
                        }
                        MessageContentItem::Refusal { refusal } => {
                            return Err(ModelError::ContentFiltered(refusal));
                        }
                    }
                }
            }
            ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                parts.push(ModelResponsePart::ToolCall(
                    ToolCallPart::new(name, ToolCallArgs::from(arguments))
                        .with_tool_call_id(call_id),
                ));
            }
            ResponseOutputItem::WebSearchCall { .. }
            | ResponseOutputItem::CodeInterpreterCall { .. }
            | ResponseOutputItem::FileSearchCall { .. }
            | ResponseOutputItem::ImageGenerationCall { .. }
            | ResponseOutputItem::McpCall { .. }
            | ResponseOutputItem::FunctionCallOutput { .. } => {}
        }
    }
    Ok(parts)
}

/// Map a serdesAI tool choice onto the wire tool choice.
pub(crate) fn tool_choice_to_wire(choice: Option<&ToolChoice>) -> Option<ResponsesToolChoice> {
    let choice = choice?;
    Some(match choice {
        ToolChoice::Auto => ResponsesToolChoice::Mode(ToolChoiceMode::Auto),
        ToolChoice::None => ResponsesToolChoice::Mode(ToolChoiceMode::None),
        ToolChoice::Required => ResponsesToolChoice::Mode(ToolChoiceMode::Required),
        ToolChoice::Specific(name) => ResponsesToolChoice::Function {
            kind: ToolChoiceFunctionTag,
            function: ToolChoiceFunction { name: name.clone() },
        },
    })
}
