//! Shared support for the responses-model integration test binaries: the
//! scripted fake-websocket server helpers plus small history and request
//! builders. Declare `mod ws_fakes_common;` from a test binary in `tests/`
//! and use the helpers that binary needs.

// Helpers are shared across test binaries; not every binary uses every one.
#![allow(dead_code)]

use futures::{SinkExt, StreamExt};
use serdes_ai_core::messages::{
    ModelRequest, ModelRequestPart, ModelResponse, SystemPromptPart, UserPromptPart,
};
use serdes_ai_models::model::ModelRequestParameters;
use serdes_ai_models::openai::responses::events::StreamEvent;
use serdes_ai_models::openai::responses::wire::{
    CreateResponseRequest, OutputContent, OutputItem, OutputItemStatus, ResponseObject,
    ResponseStatus, ResponseUsage,
};
use serdes_ai_models::openai::responses::{OpenAIResponsesModel, Transport};
use tokio::net::TcpListener;
use tokio_tungstenite::{WebSocketStream, accept_async, tungstenite::Message};

/// A user turn with a plain text prompt.
pub fn user_turn(text: &str) -> ModelRequest {
    ModelRequest::with_parts(vec![ModelRequestPart::UserPrompt(UserPromptPart::new(
        text,
    ))])
}

/// A system turn.
pub fn system_turn(text: &str) -> ModelRequest {
    ModelRequest::with_parts(vec![ModelRequestPart::SystemPrompt(SystemPromptPart::new(
        text,
    ))])
}

pub fn params() -> ModelRequestParameters {
    ModelRequestParameters::new()
}

pub fn settings() -> serdes_ai_core::ModelSettings {
    serdes_ai_core::ModelSettings::default()
}

/// Concatenated text parts of a response.
pub fn text_of(response: &serdes_ai_core::ModelResponse) -> String {
    response
        .text_parts()
        .map(|part| part.content.as_str())
        .collect()
}

/// An assistant-echo turn for extending a conversation history.
pub fn response_turn(response: ModelResponse) -> ModelRequest {
    ModelRequest::with_parts(vec![ModelRequestPart::ModelResponse(Box::new(response))])
}

/// A websocket-transport model dialing the fake server's endpoint.
pub fn ws_client(addr: std::net::SocketAddr) -> OpenAIResponsesModel {
    OpenAIResponsesModel::new("test-model", "test-key")
        .with_base_url(format!("ws://{addr}/v1/responses"))
        .with_transport(Transport::WebSocket)
}

pub type FakeWs = WebSocketStream<tokio::net::TcpStream>;

/// Accept one websocket connection.
pub async fn accept_ws(listener: &TcpListener) -> FakeWs {
    let (stream, _) = listener.accept().await.unwrap();
    accept_async(stream).await.unwrap()
}

/// Read one `response.create` frame from the client.
pub async fn read_turn(ws: &mut FakeWs) -> CreateResponseRequest {
    loop {
        let message = ws.next().await.expect("frame").expect("ws ok");
        match message {
            Message::Text(text) => {
                let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(value["type"], "response.create", "unexpected frame: {text}");
                // Codex frames are flat; everything except `type` is the
                // response payload.
                value.as_object_mut().expect("frame object").remove("type");
                return serde_json::from_value(value).unwrap();
            }
            Message::Close(_) => panic!("client closed before sending a turn"),
            _ => continue,
        }
    }
}

/// Run a minimal-but-realistic turn: item added, text delta, item done,
/// completed. The client assembles parts from the streamed item events, so
/// a bare `response.completed` would leave the folded response empty.
pub async fn send_completed_turn(ws: &mut FakeWs, id: &str, request: &CreateResponseRequest) {
    let mut response = ResponseObject::in_progress(id, 0, request.model.clone(), request);
    response.status = ResponseStatus::Completed;
    response.output = vec![OutputItem::Message {
        id: format!("msg_{id}"),
        role: "assistant".to_string(),
        status: OutputItemStatus::Completed,
        content: vec![OutputContent::OutputText {
            text: "ok".to_string(),
            annotations: Vec::new(),
        }],
    }];
    response.usage = Some(ResponseUsage {
        input_tokens: Some(1),
        output_tokens: Some(1),
        total_tokens: Some(2),
    });

    let events = vec![
        StreamEvent::OutputItemAdded {
            sequence_number: 1,
            output_index: 0,
            item: OutputItem::Message {
                id: format!("msg_{id}"),
                role: "assistant".to_string(),
                status: OutputItemStatus::InProgress,
                content: Vec::new(),
            },
        },
        StreamEvent::OutputTextDelta {
            sequence_number: 2,
            item_id: format!("msg_{id}"),
            output_index: 0,
            content_index: 0,
            delta: "ok".to_string(),
        },
        StreamEvent::OutputItemDone {
            sequence_number: 3,
            output_index: 0,
            item: response.output[0].clone(),
        },
        StreamEvent::ResponseCompleted {
            sequence_number: 4,
            response,
        },
    ];
    for event in &events {
        send_event(ws, event).await;
    }
}

pub async fn send_event(ws: &mut FakeWs, event: &StreamEvent) {
    ws.send(Message::text(serde_json::to_string(event).unwrap()))
        .await
        .unwrap();
}
