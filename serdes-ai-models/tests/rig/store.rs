//! In-memory response storage for the test rig's stateful sessions.
//!
//! The rig must remember each stored response and the conversation history
//! that produced it so clients can chain turns with `previous_response_id`,
//! and keep a per-connection cache so `store: false` turns (the codex CLI
//! default) can continue a conversation on the same socket. Both live here as
//! plain in-memory maps: this is test infrastructure, not a product surface.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serdes_ai_core::ModelRequest;
use serdes_ai_models::openai::responses::wire::ResponseObject;
use std::collections::HashMap;

/// A stored response plus the history that produced it.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    /// Response ID (`resp_...`).
    pub id: String,
    /// The response object as returned to the client.
    pub response: ResponseObject,
    /// Full conversation history up to and including this turn.
    pub history: Vec<ModelRequest>,
    /// When the response was stored.
    pub stored_at: DateTime<Utc>,
}

/// Bounded in-memory store of responses for `previous_response_id` chaining.
pub struct InMemoryResponseStore {
    entries: RwLock<HashMap<String, (u64, StoredResponse)>>,
    capacity: usize,
    counter: std::sync::atomic::AtomicU64,
}

impl InMemoryResponseStore {
    /// Create a store holding at most `capacity` responses.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            capacity: capacity.max(1),
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Fetch a stored response by ID.
    pub async fn get(&self, id: &str) -> Option<StoredResponse> {
        self.entries
            .read()
            .get(id)
            .map(|(_, stored)| stored.clone())
    }

    /// Store a response, evicting the oldest entries first when at capacity.
    ///
    /// Check and eviction share one write lock so concurrent puts cannot
    /// oversubscribe capacity.
    pub async fn put(&self, stored: StoredResponse) {
        let seq = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut entries = self.entries.write();
        while entries.len() >= self.capacity {
            let evict = entries
                .iter()
                .min_by_key(|(_, (seq, _))| *seq)
                .map(|(id, _)| id.clone());
            match evict {
                Some(id) => {
                    entries.remove(&id);
                }
                None => break,
            }
        }
        entries.insert(stored.id.clone(), (seq, stored));
    }

    /// Delete a stored response.
    pub async fn delete(&self, id: &str) {
        self.entries.write().remove(id);
    }
}

impl Default for InMemoryResponseStore {
    fn default() -> Self {
        Self::new(1024)
    }
}
/// Connection-local response cache for websocket sessions.
///
/// Holds the most recent responses of a single websocket connection so that
/// `store: false` turns can chain via `previous_response_id` without global
/// storage. Per the Open Responses websocket transport rules, a failed
/// continuation evicts the referenced ID so the client is forced to replay
/// the full input on its next attempt.
pub struct SessionResponseCache {
    entries: RwLock<HashMap<String, (u64, StoredResponse)>>,
    capacity: usize,
    counter: std::sync::atomic::AtomicU64,
}

impl SessionResponseCache {
    /// Create a session cache holding at most `capacity` responses.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            capacity: capacity.max(1),
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Fetch a cached response by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<StoredResponse> {
        self.entries
            .read()
            .get(id)
            .map(|(_, stored)| stored.clone())
    }

    /// Cache a response, evicting the oldest entry when full.
    ///
    /// Eviction order is a monotonic insertion counter, not a wall-clock
    /// timestamp, so responses stored in the same instant evict in true
    /// insertion order.
    pub fn put(&self, stored: StoredResponse) {
        let seq = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut entries = self.entries.write();
        while entries.len() >= self.capacity {
            let evict = entries
                .iter()
                .min_by_key(|(_, (seq, _))| *seq)
                .map(|(id, _)| id.clone());
            match evict {
                Some(id) => {
                    entries.remove(&id);
                }
                None => break,
            }
        }
        entries.insert(stored.id.clone(), (seq, stored));
    }

    /// Evict a response ID (used after a failed continuation turn).
    pub fn evict(&self, id: &str) {
        self.entries.write().remove(id);
    }
}

impl Default for SessionResponseCache {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_models::openai::responses::wire::{CreateResponseRequest, ResponseInput};

    fn stored(id: &str, stored_at: DateTime<Utc>) -> StoredResponse {
        StoredResponse {
            id: id.to_string(),
            response: ResponseObject::in_progress(
                id,
                0,
                "m",
                &CreateResponseRequest {
                    model: "m".into(),
                    input: ResponseInput::default(),
                    instructions: None,
                    tools: None,
                    tool_choice: None,
                    temperature: None,
                    top_p: None,
                    max_output_tokens: None,
                    stream: None,
                    background: None,
                    store: None,
                    previous_response_id: None,
                    reasoning: None,
                    parallel_tool_calls: None,
                    metadata: None,
                    user: None,
                    truncation: None,
                    include: None,
                    text: None,
                    service_tier: None,
                },
            ),
            history: Vec::new(),
            stored_at,
        }
    }

    #[tokio::test]
    async fn put_get_delete_roundtrip() {
        let store = InMemoryResponseStore::new(4);
        assert!(store.get("resp_missing").await.is_none());
        store.put(stored("resp_1", Utc::now())).await;
        assert!(store.get("resp_1").await.is_some());
        store.delete("resp_1").await;
        assert!(store.get("resp_1").await.is_none());
    }

    #[tokio::test]
    async fn capacity_evicts_oldest() {
        let store = InMemoryResponseStore::new(2);
        let base = Utc::now();
        store.put(stored("resp_a", base)).await;
        std::thread::sleep(std::time::Duration::from_millis(2));
        store
            .put(stored("resp_b", base + chrono::Duration::seconds(1)))
            .await;
        store
            .put(stored("resp_c", base + chrono::Duration::seconds(2)))
            .await;
        assert!(store.get("resp_a").await.is_none(), "oldest evicted");
        assert!(store.get("resp_b").await.is_some());
        assert!(store.get("resp_c").await.is_some());
    }

    #[test]
    fn session_cache_evicts_referenced_id() {
        let cache = SessionResponseCache::new(2);
        cache.put(stored("resp_1", Utc::now()));
        cache.put(stored("resp_2", Utc::now()));
        assert!(cache.get("resp_1").is_some());
        cache.evict("resp_1");
        assert!(cache.get("resp_1").is_none());
        cache.evict("resp_never_there"); // no-op
        assert!(cache.get("resp_2").is_some());
    }
}
