//! HTTP transport for the Open Responses model: session-chained turns and
//! SSE streaming.
//!
//! Chained HTTP turns share the websocket transport's conversation state
//! (see [`super::session`]): each conversation serializes its turns, a
//! completed turn records its response id, and the next turn sends only
//! the new input items chained via `previous_response_id`, persisted with
//! `store: true` so the server can resolve the chain. A stale
//! `previous_response_not_found` before any output escapes clears the
//! chain and replays the full input, bounded by [`MAX_ATTEMPTS`]. Error
//! envelopes surface as provider errors with the wire code, the same
//! mapping the websocket transport uses; the bare status error is reserved
//! for bodies the server did not mark with an envelope.

use std::time::Duration;

use futures::StreamExt;
use serdes_ai_core::messages::{ModelRequest, ModelResponseStreamEvent};
use serdes_ai_core::{ModelResponse, ModelSettings};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::events::{StreamEvent, failure_kind, translate};
use super::session::{Conv, MAX_ATTEMPTS, fingerprint};
use super::wire::{HttpErrorEnvelope, codes};
use super::{OpenAIResponsesModel, RequestOverlay, ResponsesApiRequest, TruncationConfig};
use crate::error::ModelError;
use crate::model::{ModelRequestParameters, StreamedResponse};

/// Compose a chained HTTP turn's request body.
///
/// The shared mapping carries the conversation plan's skip point and
/// continuation id; HTTP pins the transport fields: `store: true` on every
/// turn (including a fresh conversation's full-replay first turn, matching
/// the reference HTTP client) and the `stream` key.
fn build_chained_request(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
    skip: usize,
    previous_response_id: Option<String>,
    stream: bool,
) -> Result<ResponsesApiRequest, ModelError> {
    model.compose_request(
        messages,
        settings,
        params,
        RequestOverlay {
            skip,
            stream: Some(stream),
            store: Some(true),
            previous_response_id,
            service_tier: model.default_settings.service_tier,
            truncation: model
                .default_settings
                .truncation
                .map(|t| TruncationConfig { truncation_type: t }),
        },
    )
}

/// POST a request body to the responses endpoint with the model's auth,
/// routing, and custom headers.
pub(super) async fn post(
    model: &OpenAIResponsesModel,
    body: &ResponsesApiRequest,
    timeout: Duration,
) -> Result<reqwest::Response, ModelError> {
    let mut request = model
        .client
        .post(format!("{}/responses", model.base_url))
        .timeout(timeout);
    for (name, value) in model.request_headers() {
        request = request.header(name, value);
    }
    request
        .json(body)
        .send()
        .await
        .map_err(|e| ModelError::Connection(e.to_string()))
}

/// Send a chained turn, retrying a stale continuation before any output
/// escapes.
///
/// A `previous_response_not_found` envelope clears the recorded chain so
/// the retry replays the full input without a continuation id, bounded by
/// [`MAX_ATTEMPTS`]. Every other failure surfaces immediately: envelope
/// bodies as provider errors with the wire code, anything else as the
/// bare transport error.
async fn post_turn(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
    state: &mut Conv,
    fingerprints: &[u64],
    stream: bool,
) -> Result<reqwest::Response, ModelError> {
    let timeout = settings.timeout.unwrap_or(model.default_timeout);
    for _attempt in 0..MAX_ATTEMPTS {
        let (skip, previous) = state.plan(fingerprints, messages);
        let chained = previous.is_some();
        let body =
            build_chained_request(model, messages, settings, params, skip, previous, stream)?;
        let response = post(model, &body, timeout).await?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status().as_u16();
        let body_text = response.text().await.unwrap_or_default();
        let envelope = serde_json::from_str::<HttpErrorEnvelope>(&body_text).ok();
        if chained
            && envelope
                .as_ref()
                .is_some_and(|envelope| envelope.error.code == codes::PREVIOUS_RESPONSE_NOT_FOUND)
        {
            tracing::warn!("stale continuation; replaying the full input");
            state.previous_response_id = None;
            state.sent_fingerprints.clear();
            continue;
        }
        return Err(match envelope {
            Some(envelope) => envelope_error(&envelope, status),
            None => ModelError::http(status, body_text),
        });
    }
    Err(ModelError::Connection(
        "http turn exhausted retries".to_string(),
    ))
}

/// Map an HTTP error envelope onto the provider error form the websocket
/// transport produces, carrying the status for transport-aware policies.
pub(super) fn envelope_error(envelope: &HttpErrorEnvelope, status: u16) -> ModelError {
    ModelError::provider_with_status(
        "openai",
        envelope.error.code.clone(),
        envelope.error.message.clone(),
        failure_kind(&envelope.error.code),
        Some(status),
        None,
    )
}

/// Run one session-chained turn over HTTP and map the completed response.
pub(crate) async fn request(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<ModelResponse, ModelError> {
    let fingerprints: Vec<u64> = messages.iter().map(fingerprint).collect();
    let conv = model.conversation(messages).await;
    let mut state = conv.lock().await;

    let response = post_turn(
        model,
        messages,
        settings,
        params,
        &mut state,
        &fingerprints,
        false,
    )
    .await?;
    let resp: super::ResponsesApiResponse = response
        .json()
        .await
        .map_err(|e| ModelError::invalid_response(e.to_string()))?;
    let response_id = resp.id.clone();
    let model_response = model.process_response(resp)?;
    // The chain advances only after the response mapped cleanly, so a
    // turn that fails to produce parts never poisons the next
    // continuation.
    state.previous_response_id = Some(response_id);
    state.sent_fingerprints = fingerprints.clone();
    state.last_used = std::time::Instant::now();
    Ok(model_response)
}

/// Stream one session-chained turn over HTTP (SSE) through a channel.
pub(crate) fn stream(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
) -> Result<StreamedResponse, ModelError> {
    let (tx, rx) = mpsc::channel::<Result<ModelResponseStreamEvent, ModelError>>(64);
    let model = model.clone();
    let messages = messages.to_vec();
    let settings = settings.clone();
    let params = params.clone();

    tokio::spawn(async move {
        let result = stream_events(&model, &messages, &settings, &params, &tx).await;
        if let Err(error) = result {
            // A failure after events escaped still reaches the caller as
            // an error item; a failure before that is the only item.
            let _ = tx.send(Err(error)).await;
        }
    });

    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Read the SSE body of one committed turn, translating wire events into
/// the channel.
///
/// The request is established before any event escapes (see [`post_turn`]);
/// once the body is being read there is no further replay. The
/// terminal-event contract is enforced: a turn ends on exactly one
/// terminal `StreamComplete` or on a surfaced failure, so an SSE `[DONE]`
/// sentinel without a terminal event, or a body that ends without one, is
/// a contract violation and surfaces as an error.
async fn stream_events(
    model: &OpenAIResponsesModel,
    messages: &[ModelRequest],
    settings: &ModelSettings,
    params: &ModelRequestParameters,
    tx: &mpsc::Sender<Result<ModelResponseStreamEvent, ModelError>>,
) -> Result<(), ModelError> {
    let fingerprints: Vec<u64> = messages.iter().map(fingerprint).collect();
    let conv = model.conversation(messages).await;
    let mut state = conv.lock().await;

    let response = post_turn(
        model,
        messages,
        settings,
        params,
        &mut state,
        &fingerprints,
        true,
    )
    .await?;

    let mut byte_stream = response.bytes_stream();
    let mut buffer = String::new();
    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.map_err(|e| ModelError::Connection(e.to_string()))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(newline) = buffer.find('\n') {
            let line: String = buffer.drain(..=newline).collect();
            let payload = line.trim_end_matches(['\n', '\r']);
            let Some(payload) = payload.strip_prefix("data: ") else {
                continue;
            };
            if payload == "[DONE]" {
                return Err(ModelError::incomplete_stream(
                    "sse stream sent [DONE] without a terminal response.completed/incomplete/failed event",
                ));
            }
            let event = match super::events::decode(payload)? {
                super::events::DataEvent::Event(event) => *event,
                super::events::DataEvent::Error(envelope) => {
                    return Err(envelope_error(
                        &HttpErrorEnvelope {
                            error: envelope.error,
                        },
                        envelope.status_code,
                    ));
                }
            };
            // Record the continuation before translation consumes the
            // event; a failed response never advances the chain.
            if let StreamEvent::ResponseCompleted { response, .. }
            | StreamEvent::ResponseIncomplete { response, .. } = &event
            {
                state.previous_response_id = Some(response.id.clone());
                state.sent_fingerprints = fingerprints.clone();
            }
            for translated in translate(event) {
                match translated {
                    Ok(event) => {
                        let terminal = matches!(event, ModelResponseStreamEvent::StreamComplete(_));
                        tx.send(Ok(event))
                            .await
                            .map_err(|_| ModelError::Cancelled)?;
                        if terminal {
                            state.last_used = std::time::Instant::now();
                            return Ok(());
                        }
                    }
                    Err(error) => {
                        // A failed turn surfaces as an error item instead of
                        // a synthetic completion; the stream ends there.
                        let _ = tx.send(Err(error)).await;
                        return Ok(());
                    }
                }
            }
        }
    }

    Err(ModelError::incomplete_stream(
        "sse stream ended without a terminal event",
    ))
}
