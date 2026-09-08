//! Conversation-keyed session state for the responses model.
//!
//! Lookup uses fingerprints of leading system-only requests and the first
//! request containing other material. Shared system prompts alone do not
//! identify a conversation. This is a bounded content-based contract:
//! callers must preserve that prefix and must not interleave independent
//! histories with the same prefix through one model. Use separate model
//! instances (not clones) for those histories. Equal content cannot prove
//! identity. Full sent-prefix checks reject edits and identical-history
//! replays reset the chain. Websockets keep a socket per lookup key; HTTP
//! records completed response ids with `store: true`.
// The EventSink helpers serve only the websocket transport; with that
// feature compiled out they are unreachable, which is expected, not dead
// code. The conversation state itself serves both transports.
#![cfg_attr(not(feature = "responses-ws"), allow(dead_code))]

use crate::error::ModelError;
use async_trait::async_trait;
use serdes_ai_core::messages::{ModelRequest, ModelRequestPart, ModelResponseStreamEvent};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc};

/// Transport used to reach the endpoint.
///
/// HTTP is the default so the unified model's behavior is unchanged unless
/// the websocket transport is selected explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    /// HTTP (`https://`/`http://`): session chaining via `store: true` and
    /// `previous_response_id` on each `POST`.
    #[default]
    Http,
    /// Websocket (`wss://`/`ws://`): session-stateful turns with
    /// `store: false` and delta-only input. This is the transport the codex
    /// CLI and Open Responses servers are designed around.
    WebSocket,
}

impl Transport {
    /// Infer the transport from a URL scheme.
    #[must_use]
    pub fn from_url(url: &str) -> Self {
        if url.starts_with("http://") || url.starts_with("https://") {
            Self::Http
        } else {
            Self::WebSocket
        }
    }
}

/// How many times a turn may be retried internally (continuation replay
/// after `previous_response_not_found`, reconnect after a connection
/// failure or the server's connection lifetime limit) before the error
/// surfaces.
pub(crate) const MAX_ATTEMPTS: usize = 3;

/// How long a conversation may sit untouched before it is evicted (and
/// its socket, if any, closed) at the next conversation lookup. The
/// default for `OpenAIResponsesModel::with_conversation_idle_ttl`.
pub(crate) const CONVERSATION_IDLE_TTL: Duration = Duration::from_secs(5 * 60);

/// Connection-local state for one conversation.
///
/// Conversations use the initial prefix described in this module.
/// The websocket variant keeps the socket
/// alive across the conversation's turns; continuation state
/// (`previous_response_id`, the requests already sent) lives here, which is
/// what makes delta-only continuation turns possible. Turns of one
/// conversation are sequential (the protocol has no way to match
/// interleaved responses) while different conversations run concurrently,
/// each on its own socket.
pub(crate) struct Conv {
    /// Live socket for this conversation, connected lazily on the first
    /// websocket turn.
    #[cfg(feature = "responses-ws")]
    pub(crate) socket: Option<serdes_ai_streaming::websocket::WebSocketStream>,
    pub(crate) previous_response_id: Option<String>,
    /// Fingerprints of the requests the server already holds, in order. A
    /// turn's history must extend this sequence exactly; anything else
    /// voids the chain and forces a full replay.
    pub(crate) sent_fingerprints: Vec<u64>,
    /// Last turn activity, refreshed at planning and completion; drives lazy
    /// idle eviction.
    pub(crate) last_used: Instant,
}

impl std::fmt::Debug for Conv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut state = f.debug_struct("Conv");
        #[cfg(feature = "responses-ws")]
        state.field("connected", &self.socket.is_some());
        state
            .field("previous_response_id", &self.previous_response_id)
            .field("sent_fingerprints", &self.sent_fingerprints)
            .field("idle_for", &self.last_used.elapsed())
            .finish()
    }
}

impl Conv {
    /// Align the recorded chain with the incoming history and compute this
    /// attempt's skip point and continuation id.
    ///
    /// The recorded fingerprints must be a prefix of `fingerprints`: a
    /// conversation may only extend the history the server already holds.
    /// When the incoming history does not extend it (same first request,
    /// mutated middle, or truncated), the chain is void: it is cleared here
    /// so the turn resends the full input without a continuation id.
    pub(crate) fn plan(
        &mut self,
        fingerprints: &[u64],
        messages: &[ModelRequest],
    ) -> (usize, Option<String>) {
        self.last_used = Instant::now();
        // A continuation must ADD material to the history the server
        // holds. When the incoming history equals the recorded sent prefix
        // exactly (zero new items), the caller is replaying the same
        // prompt rather than continuing the conversation; chaining it
        // would send an empty input against the old continuation id. That
        // replay is treated as a chain reset instead: the full input goes
        // out with no continuation id, and the conversation re-chains from
        // the replay's response.
        if self.previous_response_id.is_some() && self.sent_fingerprints == fingerprints {
            tracing::debug!("history repeats the sent chain; restarting the conversation");
            self.previous_response_id = None;
            self.sent_fingerprints.clear();
        }
        let extends = self.sent_fingerprints.len() <= fingerprints.len()
            && self
                .sent_fingerprints
                .iter()
                .zip(fingerprints)
                .all(|(sent, incoming)| sent == incoming);
        if !extends {
            tracing::debug!(
                sent = self.sent_fingerprints.len(),
                incoming = fingerprints.len(),
                "history diverged from the sent chain; restarting the conversation"
            );
            self.previous_response_id = None;
            self.sent_fingerprints.clear();
        }
        let chained = self.previous_response_id.is_some();
        let skip = if chained {
            continuation_skip(messages, self.sent_fingerprints.len())
        } else {
            0
        };
        // Invariant: chained was derived from the same Option above.
        let previous = chained.then(|| self.previous_response_id.clone().expect("checked"));
        (skip, previous)
    }

    /// Take the live socket, if any, so the caller can close it cleanly.
    #[cfg(feature = "responses-ws")]
    pub(crate) fn take_socket(
        &mut self,
    ) -> Option<serdes_ai_streaming::websocket::WebSocketStream> {
        self.socket.take()
    }
}

/// Teardown belongs to the socket owner, not to a model handle checking
/// Arc counts. This also closes sockets when the last clones drop concurrently.
/// Without a current runtime a close frame cannot be sent asynchronously.
#[cfg(feature = "responses-ws")]
impl Drop for Conv {
    fn drop(&mut self) {
        if let Some(mut socket) = self.socket.take() {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    close_socket(&mut socket).await;
                });
            }
        }
    }
}

/// A conversation whose turns serialize on their own lock.
pub(crate) type SharedConv = Arc<Mutex<Conv>>;

/// Close a websocket with the proper handshake so the peer observes a
/// Close frame instead of a bare connection reset, then leave the stream
/// to drop.
///
/// Best-effort: sockets are usually closed because they already failed or
/// the peer went away, so a rejected handshake is logged, not propagated.
#[cfg(feature = "responses-ws")]
pub(crate) async fn close_socket(socket: &mut serdes_ai_streaming::websocket::WebSocketStream) {
    if let Err(error) = socket.close().await {
        tracing::debug!(%error, "websocket close handshake failed");
    }
}

/// Fingerprint a request for in-process conversation keying.
///
/// The hash covers the request kind and the serialized request parts, in
/// that order. `DefaultHasher` is not stable across processes or compiler
/// releases, so fingerprints are in-process only: they key in-memory
/// conversation state and are never persisted nor sent on the wire.
/// Serialization of the parts is deterministic for a given value (serde
/// writes fields in declaration order), and infallible for these
/// serde-derived types, so equal requests always hash equally within a
/// process.
pub(crate) fn fingerprint(request: &ModelRequest) -> u64 {
    let mut hasher = DefaultHasher::new();
    request.kind.hash(&mut hasher);
    serde_json::to_string(&request.parts)
        .expect("request parts are serde types and always serialize")
        .hash(&mut hasher);
    hasher.finish()
}

/// Advance the continuation skip point past the assistant echo.
///
/// After a completed turn the caller appends the response to its local
/// history; with `previous_response_id` chaining the server already has that
/// output, so trailing model-response requests are not re-sent. Only used on
/// chained turns: a fresh session needs the full replay, old assistant
/// output included.
fn continuation_skip(messages: &[ModelRequest], sent: usize) -> usize {
    let mut skip = sent;
    while skip < messages.len()
        && messages[skip]
            .parts
            .iter()
            .all(|part| matches!(part, ModelRequestPart::ModelResponse(_)))
    {
        skip += 1;
    }
    skip
}

/// Where a websocket turn's event stream goes.
///
/// Non-streaming turns collect events for folding; streaming turns forward
/// them through a channel. The sink is async so streaming callers get
/// backpressure instead of dropped events.
#[async_trait]
pub(crate) trait EventSink: Send {
    async fn send(&mut self, event: ModelResponseStreamEvent) -> Result<(), ModelError>;
}

/// Collects events for later folding into a `ModelResponse`.
pub(crate) struct CollectSink(pub(crate) Vec<ModelResponseStreamEvent>);

#[async_trait]
impl EventSink for CollectSink {
    async fn send(&mut self, event: ModelResponseStreamEvent) -> Result<(), ModelError> {
        self.0.push(event);
        Ok(())
    }
}

/// Forwards events to a streaming caller.
pub(crate) struct ChannelSink<'a>(
    pub(crate) &'a mpsc::Sender<Result<ModelResponseStreamEvent, ModelError>>,
);

#[async_trait]
impl EventSink for ChannelSink<'_> {
    async fn send(&mut self, event: ModelResponseStreamEvent) -> Result<(), ModelError> {
        self.0
            .send(Ok(event))
            .await
            .map_err(|_| ModelError::Cancelled)
    }
}

impl super::OpenAIResponsesModel {
    /// Look up or create the conversation state for this history.
    ///
    /// Conversations use the initial prefix described in this module.
    /// The map lock guards lookup, insert, and the eviction scan
    /// (no await is performed under it);
    /// turn serialization happens on the conversation's own lock, so
    /// different conversations never wait on each other.
    pub(crate) async fn conversation(&self, messages: &[ModelRequest]) -> SharedConv {
        let prefix_len = messages
            .iter()
            .position(|request| {
                request
                    .parts
                    .iter()
                    .any(|part| !matches!(part, ModelRequestPart::SystemPrompt(_)))
            })
            .map_or(messages.len(), |index| index + 1);
        let key: Vec<u64> = messages[..prefix_len].iter().map(fingerprint).collect();
        self.evict_idle_conversations().await;
        self.conversations
            .lock()
            .expect("conversations map lock poisoned; it is only held for lookup/insert")
            .entry(key)
            .or_insert_with(|| {
                Arc::new(Mutex::new(Conv {
                    #[cfg(feature = "responses-ws")]
                    socket: None,
                    previous_response_id: None,
                    sent_fingerprints: Vec::new(),
                    last_used: Instant::now(),
                }))
            })
            .clone()
    }

    /// Drop conversations idle past the configured TTL, closing their
    /// sockets with a proper handshake so the peer sees a Close frame
    /// rather than a connection reset.
    ///
    /// Lazy by design: each lookup scans the live conversations instead of
    /// running a background task. Both locked turns and handles reserved by
    /// queued turns prevent eviction. Lookup and the ownership check share
    /// the map lock, so an evicted conversation has no remaining caller.
    /// Its socket closes after releasing the map lock.
    async fn evict_idle_conversations(&self) {
        let evicted: Vec<SharedConv> = {
            let mut conversations = self
                .conversations
                .lock()
                .expect("conversations map lock poisoned; it is only held for lookup/insert");
            let mut evicted = Vec::new();
            conversations.retain(|_, conv| match conv.try_lock() {
                // A looked-up or queued turn owns a handle before locking it.
                Ok(_) if Arc::strong_count(conv) != 1 => true,
                Ok(state) => {
                    let idle = state.last_used.elapsed() > self.conversation_idle_ttl;
                    if idle {
                        evicted.push(Arc::clone(conv));
                    }
                    !idle
                }
                Err(_mid_turn) => true,
            });
            evicted
        };
        #[cfg(feature = "responses-ws")]
        for conv in evicted {
            let mut state = conv.lock().await;
            if let Some(mut socket) = state.take_socket() {
                close_socket(&mut socket).await;
            }
        }
        // Without the websocket transport there are no sockets to close;
        // the evicted conversation state is simply dropped.
        #[cfg(not(feature = "responses-ws"))]
        drop(evicted);
    }
}

#[cfg(all(test, feature = "responses-ws"))]
mod session_state_tests {
    use super::{Conv, Transport, fingerprint};
    use crate::model::ModelRequestParameters;
    use crate::openai::responses::OpenAIResponsesModel;
    use crate::openai::responses::ws::build_request;
    use serdes_ai_core::ModelSettings;
    use serdes_ai_core::messages::{ModelRequest, ModelRequestPart, UserPromptPart};
    use std::time::Instant;

    fn model() -> OpenAIResponsesModel {
        OpenAIResponsesModel::new("gpt-5.6-luna", "test-key").with_transport(Transport::WebSocket)
    }

    fn user_request(text: &str) -> ModelRequest {
        ModelRequest::with_parts(vec![ModelRequestPart::UserPrompt(UserPromptPart::new(
            text,
        ))])
    }

    #[test]
    fn fingerprints_match_identical_requests_and_split_different_ones() {
        // Equality is by value: a conversation's history re-presents the
        // same request values on every turn (parts carry their creation
        // timestamp), so clones fingerprint alike.
        let request = user_request("a");
        assert_eq!(fingerprint(&request), fingerprint(&request.clone()));
        assert_ne!(
            fingerprint(&request),
            fingerprint(&user_request("b")),
            "different requests must not share a fingerprint"
        );
    }

    #[test]
    fn plan_keeps_the_chain_only_when_history_extends_the_sent_prefix() {
        let mut conv = Conv {
            socket: None,
            previous_response_id: Some("resp_1".to_string()),
            sent_fingerprints: vec![1, 2, 3],
            last_used: Instant::now(),
        };

        // An extension chains and skips past the verified prefix.
        let (skip, previous) = conv.plan(&[1, 2, 3, 4], &[]);
        assert_eq!(previous.as_deref(), Some("resp_1"));
        assert_eq!(skip, 3);
        assert_eq!(conv.sent_fingerprints, vec![1, 2, 3]);

        // A truncated or diverging history voids the chain: full replay,
        // no continuation id.
        let mut conv = Conv {
            socket: None,
            previous_response_id: Some("resp_1".to_string()),
            sent_fingerprints: vec![1, 2, 3],
            last_used: Instant::now(),
        };
        let (skip, previous) = conv.plan(&[1, 2, 9], &[]);
        assert_eq!(previous, None);
        assert_eq!(skip, 0);
        assert!(conv.previous_response_id.is_none());
        assert!(conv.sent_fingerprints.is_empty());
    }

    #[test]
    fn an_empty_turn_serializes_input_as_a_list() {
        // A caller that hands the model an empty history gets an empty
        // input list. The API rejects an empty string here with "Input
        // must be a list", so the empty case has to stay an array. (A
        // repeated history is not an empty turn anymore: it resets the
        // chain and re-sends the full input — see `Conv::plan`.)
        let messages: Vec<ModelRequest> = Vec::new();
        let request = build_request(
            &model(),
            &messages,
            &ModelSettings::default(),
            &ModelRequestParameters::default(),
            0,
            Some("resp_1".to_string()),
        )
        .expect("request");

        let body = serde_json::to_value(&request).expect("serialize");
        assert!(
            body["input"].is_array(),
            "input must be a list when empty, got {}",
            body["input"]
        );
    }

    #[test]
    fn plan_resets_the_chain_when_history_repeats_the_sent_prefix() {
        // Re-sending exactly the delivered history is a replay of the same
        // prompt, not a continuation: a continuation must add material.
        // The chain resets so the full input goes out without a
        // continuation id.
        let mut conv = Conv {
            socket: None,
            previous_response_id: Some("resp_1".to_string()),
            sent_fingerprints: vec![1, 2, 3],
            last_used: Instant::now(),
        };
        let (skip, previous) = conv.plan(&[1, 2, 3], &[]);
        assert_eq!(previous, None, "a replayed history must not chain");
        assert_eq!(skip, 0, "a replayed history re-sends the full input");
        assert!(conv.previous_response_id.is_none());
        assert!(conv.sent_fingerprints.is_empty());

        // Once the state records a fresh turn, extending the history
        // chains again as usual.
        conv.previous_response_id = Some("resp_2".to_string());
        conv.sent_fingerprints = vec![1, 2, 3];
        let (skip, previous) = conv.plan(&[1, 2, 3, 4], &[]);
        assert_eq!(previous.as_deref(), Some("resp_2"));
        assert_eq!(skip, 3);
    }

    #[tokio::test]
    async fn idle_eviction_preserves_a_reserved_turn_until_its_handle_is_released() {
        let model = model().with_conversation_idle_ttl(std::time::Duration::ZERO);
        let history = vec![user_request("reserved")];
        let reserved = model.conversation(&history).await;
        reserved.lock().await.previous_response_id = Some("resp_reserved".into());
        // A turn owns a handle between lookup and acquiring its turn lock.
        let next = model.conversation(&history).await;
        assert_eq!(
            next.lock().await.previous_response_id.as_deref(),
            Some("resp_reserved")
        );
        drop(next);
        drop(reserved);
        let expired = model.conversation(&history).await;
        assert!(expired.lock().await.previous_response_id.is_none());
    }

    #[test]
    fn fingerprints_include_the_request_kind() {
        // The kind participates in the hash: the same parts under a
        // different kind must not collide in the conversation map.
        let request = user_request("a");
        let mut relabeled = request.clone();
        relabeled.kind = "other".to_string();
        assert_ne!(
            fingerprint(&request),
            fingerprint(&relabeled),
            "the kind must be part of the fingerprint"
        );
    }
}
