//! Application-owned context policy and durable observation, without storage coupling.
use crate::{AgentRunError, HistoryProcessor, RunContext, RunUsage};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serdes_ai_core::{ModelRequest, ModelResponse, ModelSettings};
use serdes_ai_models::{Model, ModelRequestParameters};
use std::sync::Arc;

/// Serializable boundary. Restoring this data never executes tools automatically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CheckpointBoundary {
    /// Canonical context is prepared; no request has started.
    BeforeRequest,
    /// Periodic partial snapshot; not a safe automatic replay boundary.
    Partial,
    /// Response received; no tools from this response have started.
    AfterResponse,
    /// Completed tool batch has been appended to active history.
    AfterTools,
    /// Run has finished normally (including token-limited output).
    Terminal,
    /// Execution was interrupted; pending tool results are not committed.
    Cancelled,
    /// Consumer detached before a committed terminal boundary.
    ConsumerDetached,
    /// Output parsing/validation exhausted its retry budget.
    ValidationFailed,
    /// Execution failed. Partial response may be present.
    Failed,
    /// Typed provider failure without embedding raw provider error text.
    ModelFailed(serdes_ai_core::ModelFailureKind),
}

/// Versioned snapshot, not an automatic side-effect replay command.
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    /// Currently 1. Consumers must reject unknown versions.
    pub version: u32,
    /// Stable run identifier.
    pub run_id: String,
    /// Request iteration.
    pub step: u32,
    /// Safe observation boundary.
    pub boundary: CheckpointBoundary,
    /// Canonical active context, not an ever-growing immutable archive.
    pub messages: Vec<ModelRequest>,
    /// Current response, including partial text/thinking/tool arguments on failure.
    pub response: Option<ModelResponse>,
    /// Aggregate usage. Per-response unknown usage stays None in response.
    pub usage: RunUsage,
}
impl std::fmt::Debug for AgentCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentCheckpoint")
            .field("version", &self.version)
            .field("step", &self.step)
            .field("boundary", &self.boundary)
            .field("payload", &"<redacted>")
            .finish()
    }
}
impl AgentCheckpoint {
    /// Decode a supported snapshot. This does not resume execution.
    pub fn from_json(json: &str) -> Result<Self, AgentRunError> {
        let checkpoint: Self = serde_json::from_str(json)?;
        if checkpoint.version != 1 {
            return Err(AgentRunError::Checkpoint(
                "unsupported checkpoint version".into(),
            ));
        }
        Ok(checkpoint)
    }
}

/// Application persistence hook. Completion is awaited before further side effects.
/// Implementations own transactions, encryption, retention and idempotency.
#[async_trait]
pub trait CheckpointSink: Send + Sync {
    /// Optional partial-stream snapshot interval. None disables periodic snapshots.
    /// Intervals below 10ms are clamped to 10ms to bound overhead.
    fn partial_interval(&self) -> Option<std::time::Duration> {
        None
    }
    /// Maximum wait per save. Timeout stops execution; the application must
    /// reconcile a potentially committed write. Cancellation does not interrupt save.
    fn save_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    /// Persist a snapshot or stop the run. No automatic retry is performed.
    async fn save(&self, checkpoint: &AgentCheckpoint) -> Result<(), String>;
}

/// Inputs for an application tokenizer/summary policy. An independent summary
/// model may be owned by the policy; the framework never selects one implicitly.
pub struct ContextPolicyInput<'a, Deps> {
    /// Dependencies and run identity.
    pub context: &'a RunContext<Deps>,
    /// Actual request settings.
    pub settings: &'a ModelSettings,
    /// Tools and output schema included in this request.
    pub parameters: &'a ModelRequestParameters,
    /// Actual native model, including profile/context window.
    pub model: &'a dyn Model,
}
/// Fallible asynchronous context policy. Preserve native blocks and tool pairs.
#[async_trait]
pub trait ContextPolicy<Deps>: Send + Sync {
    /// Return the replacement canonical active history. Failure stops by default.
    async fn prepare(
        &self,
        input: ContextPolicyInput<'_, Deps>,
        messages: Vec<ModelRequest>,
    ) -> Result<Vec<ModelRequest>, String>;
}
/// Explicit policy failure behavior; no implicit summarize-to-truncate fallback.
#[derive(Debug, Clone, Copy, Default)]
pub enum ContextFailurePolicy {
    /// Stop before requesting the model.
    #[default]
    Stop,
    /// Explicitly retain the pre-policy history (it may exceed model budget).
    KeepHistory,
}

pub(crate) async fn prepare<Deps: Send + Sync>(
    processors: &[Arc<dyn HistoryProcessor<Deps>>],
    policy: Option<&Arc<dyn ContextPolicy<Deps>>>,
    failure: ContextFailurePolicy,
    input: ContextPolicyInput<'_, Deps>,
    messages: &mut Vec<ModelRequest>,
) -> Result<(), AgentRunError> {
    let mut candidate = messages.clone();
    for processor in processors {
        candidate = processor.process(input.context, candidate).await;
    }
    if let Some(policy) = policy {
        match policy.prepare(input, candidate.clone()).await {
            Ok(prepared) => candidate = prepared,
            Err(error) => match failure {
                ContextFailurePolicy::Stop => return Err(AgentRunError::ContextPolicy(error)),
                ContextFailurePolicy::KeepHistory => {}
            },
        }
    }
    if policy.is_some() || !processors.is_empty() {
        validate_tool_pairs(&candidate)?;
    }
    *messages = candidate;
    Ok(())
}

tokio::task_local! {
    pub(crate) static SAVE_SCOPE: (Arc<std::sync::atomic::AtomicBool>, tokio_util::sync::CancellationToken);
}

pub(crate) async fn save(
    sink: Option<&Arc<dyn CheckpointSink>>,
    run_id: &str,
    step: u32,
    boundary: CheckpointBoundary,
    messages: &[ModelRequest],
    response: Option<&ModelResponse>,
    usage: &RunUsage,
) -> Result<(), AgentRunError> {
    let terminal = boundary == CheckpointBoundary::Terminal;
    let scope = SAVE_SCOPE.try_with(Clone::clone).ok();
    if let Some((saving, _)) = &scope {
        saving.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    if let Some(sink) = sink {
        let checkpoint = AgentCheckpoint {
            version: 1,
            run_id: run_id.into(),
            step,
            boundary,
            messages: messages.to_vec(),
            response: response.cloned(),
            usage: usage.clone(),
        };
        tokio::time::timeout(sink.save_timeout(), sink.save(&checkpoint))
            .await
            .map_err(|_| {
                AgentRunError::Checkpoint(
                    "checkpoint deadline exceeded; reconcile storage before resume".into(),
                )
            })?
            .map_err(AgentRunError::Checkpoint)?;
    }
    if let Some((saving, token)) = scope {
        saving.store(false, std::sync::atomic::Ordering::SeqCst);
        if token.is_cancelled() && !terminal {
            return Err(AgentRunError::Cancelled);
        }
    }
    Ok(())
}

// Refuse a policy that leaves a tool result without its native call. Never repair
// by inventing calls or arguments. Policies should compact whole interaction groups.
fn validate_tool_pairs(messages: &[ModelRequest]) -> Result<(), AgentRunError> {
    use serdes_ai_core::{ModelRequestPart, ModelResponsePart};
    let mut calls = std::collections::HashSet::new();
    for message in messages {
        for part in &message.parts {
            match part {
                ModelRequestPart::ModelResponse(response) => {
                    for part in &response.parts {
                        if let ModelResponsePart::ToolCall(call) = part {
                            if let Some(id) = &call.tool_call_id {
                                calls.insert(id.clone());
                            }
                        }
                    }
                }
                ModelRequestPart::ToolReturn(result) => {
                    if let Some(id) = &result.tool_call_id {
                        if !calls.remove(id) {
                            return Err(AgentRunError::ContextPolicy(
                                "orphan tool result after context policy".into(),
                            ));
                        }
                    }
                }
                ModelRequestPart::RetryPrompt(retry) => {
                    if let Some(id) = &retry.tool_call_id {
                        calls.remove(id);
                    }
                }
                _ => {}
            }
        }
    }
    if !calls.is_empty() {
        return Err(AgentRunError::ContextPolicy(
            "unresolved tool calls in prepared history; reconcile before resuming".into(),
        ));
    }
    Ok(())
}

pub(crate) fn partial_timer(sink: Option<&Arc<dyn CheckpointSink>>) -> tokio::time::Interval {
    let interval = sink
        .and_then(|sink| sink.partial_interval())
        .unwrap_or(std::time::Duration::from_secs(86400 * 365))
        .max(std::time::Duration::from_millis(10));
    let mut timer = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    timer
}

/// Update the checkpoint copy before awaiting downstream event delivery.
pub(crate) fn apply_partial(
    response: &mut ModelResponse,
    event: &serdes_ai_core::ModelResponseStreamEvent,
) {
    use serdes_ai_core::ModelResponseStreamEvent as Event;
    match event {
        Event::PartStart(start) => {
            if start.index == response.parts.len() {
                response.parts.push(start.part.clone());
            } else if let Some(part) = response.parts.get_mut(start.index) {
                *part = start.part.clone();
            }
        }
        Event::PartDelta(delta) => {
            if let Some(part) = response.parts.get_mut(delta.index) {
                let _ = delta.delta.apply(part);
            }
        }
        Event::StreamComplete(complete) => {
            response.finish_reason = Some(complete.finish_reason);
            if let Some(metadata) = &complete.metadata {
                metadata.apply(response);
            }
        }
        _ => {}
    }
}
