//! Deferred tool execution for approval flows.
//!
//! This module provides types for handling tool calls that require
//! human approval before execution.

use serde::{Deserialize, Serialize};

use crate::ToolReturn;

/// A tool call that was deferred for approval.
///
/// When a tool requires approval (via `ToolError::ApprovalRequired`),
/// the call is captured as a `DeferredToolCall` for later processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferredToolCall {
    /// Name of the tool.
    pub tool_name: String,
    /// Arguments for the tool call.
    pub args: serde_json::Value,
    /// Tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl DeferredToolCall {
    /// Create a new deferred tool call.
    #[must_use]
    pub fn new(
        tool_name: impl Into<String>,
        args: serde_json::Value,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            args,
            tool_call_id: None,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Create an approved decision.
    #[must_use]
    pub fn approve(&self) -> DeferredToolDecision {
        DeferredToolDecision::Approved
    }

    /// Create a denied decision.
    #[must_use]
    pub fn deny(&self, message: impl Into<String>) -> DeferredToolDecision {
        DeferredToolDecision::Denied(message.into())
    }

    /// Create a custom result decision.
    #[must_use]
    pub fn with_result(&self, result: ToolReturn) -> DeferredToolDecision {
        DeferredToolDecision::CustomResult(result)
    }
}

/// Collection of deferred tool calls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeferredToolRequests {
    /// The deferred calls.
    pub calls: Vec<DeferredToolCall>,
}

impl DeferredToolRequests {
    /// Create a new empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Add a deferred call.
    pub fn add(&mut self, call: DeferredToolCall) {
        self.calls.push(call);
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Get the number of deferred calls.
    #[must_use]
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Get a call by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&DeferredToolCall> {
        self.calls.get(index)
    }

    /// Iterate over the calls.
    pub fn iter(&self) -> impl Iterator<Item = &DeferredToolCall> {
        self.calls.iter()
    }

    /// Get calls by tool name.
    #[must_use]
    pub fn by_tool(&self, name: &str) -> Vec<&DeferredToolCall> {
        self.calls.iter().filter(|c| c.tool_name == name).collect()
    }

    /// Clear all calls.
    pub fn clear(&mut self) {
        self.calls.clear();
    }

    /// Approve all calls.
    #[must_use]
    pub fn approve_all(&self) -> DeferredToolDecisions {
        DeferredToolDecisions {
            decisions: self.calls.iter().map(|_| DeferredToolDecision::Approved).collect(),
        }
    }

    /// Deny all calls with a message.
    #[must_use]
    pub fn deny_all(&self, message: impl Into<String>) -> DeferredToolDecisions {
        let msg = message.into();
        DeferredToolDecisions {
            decisions: self.calls.iter().map(|_| DeferredToolDecision::Denied(msg.clone())).collect(),
        }
    }
}

impl FromIterator<DeferredToolCall> for DeferredToolRequests {
    fn from_iter<T: IntoIterator<Item = DeferredToolCall>>(iter: T) -> Self {
        Self {
            calls: iter.into_iter().collect(),
        }
    }
}

/// Decision about a deferred tool call.
#[derive(Debug, Clone)]
pub enum DeferredToolDecision {
    /// Approve the tool call.
    Approved,
    /// Deny with a message to send back to the model.
    Denied(String),
    /// Provide a custom result.
    CustomResult(ToolReturn),
}

impl DeferredToolDecision {
    /// Check if approved.
    #[must_use]
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// Check if denied.
    #[must_use]
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied(_))
    }

    /// Check if custom result.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::CustomResult(_))
    }

    /// Get the denial message if denied.
    #[must_use]
    pub fn denial_message(&self) -> Option<&str> {
        match self {
            Self::Denied(msg) => Some(msg),
            _ => None,
        }
    }
}

/// Collection of decisions for deferred tools.
#[derive(Debug, Clone, Default)]
pub struct DeferredToolDecisions {
    /// The decisions.
    pub decisions: Vec<DeferredToolDecision>,
}

impl DeferredToolDecisions {
    /// Create a new empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self { decisions: Vec::new() }
    }

    /// Add a decision.
    pub fn add(&mut self, decision: DeferredToolDecision) {
        self.decisions.push(decision);
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Get the number of decisions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Check if all are approved.
    #[must_use]
    pub fn all_approved(&self) -> bool {
        self.decisions.iter().all(|d| d.is_approved())
    }

    /// Check if any are denied.
    #[must_use]
    pub fn any_denied(&self) -> bool {
        self.decisions.iter().any(|d| d.is_denied())
    }
}

impl FromIterator<DeferredToolDecision> for DeferredToolDecisions {
    fn from_iter<T: IntoIterator<Item = DeferredToolDecision>>(iter: T) -> Self {
        Self {
            decisions: iter.into_iter().collect(),
        }
    }
}

/// Result for a single deferred tool.
#[derive(Debug, Clone)]
pub struct DeferredToolResult {
    /// Tool call ID.
    pub tool_call_id: Option<String>,
    /// The result.
    pub result: ToolReturn,
}

impl DeferredToolResult {
    /// Create a new result.
    #[must_use]
    pub fn new(result: ToolReturn) -> Self {
        Self {
            tool_call_id: None,
            result,
        }
    }

    /// Set the tool call ID.
    #[must_use]
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Create an approved result.
    #[must_use]
    pub fn approved() -> Self {
        Self::new(ToolReturn::text("Tool execution approved"))
    }

    /// Create a denied result.
    #[must_use]
    pub fn denied(message: impl Into<String>) -> Self {
        Self::new(ToolReturn::error(message))
    }
}

/// Results for all deferred tools.
#[derive(Debug, Clone, Default)]
pub struct DeferredToolResults {
    /// The results.
    pub results: Vec<DeferredToolResult>,
}

impl DeferredToolResults {
    /// Create a new empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Add a result.
    pub fn add(&mut self, result: DeferredToolResult) {
        self.results.push(result);
    }

    /// Create a single approved result.
    #[must_use]
    pub fn approved(id: Option<String>) -> Self {
        let mut result = DeferredToolResult::approved();
        if let Some(id) = id {
            result = result.with_tool_call_id(id);
        }
        Self {
            results: vec![result],
        }
    }

    /// Create a single denied result.
    #[must_use]
    pub fn denied(id: Option<String>, message: impl Into<String>) -> Self {
        let mut result = DeferredToolResult::denied(message);
        if let Some(id) = id {
            result = result.with_tool_call_id(id);
        }
        Self {
            results: vec![result],
        }
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }
}

impl FromIterator<DeferredToolResult> for DeferredToolResults {
    fn from_iter<T: IntoIterator<Item = DeferredToolResult>>(iter: T) -> Self {
        Self {
            results: iter.into_iter().collect(),
        }
    }
}

/// Marker type for approved tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolApproved;

/// Marker type for denied tools.
#[derive(Debug, Clone)]
pub struct ToolDenied {
    /// The denial message.
    pub message: String,
}

impl ToolDenied {
    /// Create a new denial.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Trait for types that can handle tool approval requests.
#[allow(async_fn_in_trait)]
pub trait ToolApprover {
    /// Handle an approval request for a tool call.
    ///
    /// Returns the decision for this tool call.
    async fn approve(&self, call: &DeferredToolCall) -> DeferredToolDecision;
}

/// Auto-approve all tool calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoApprover;

impl ToolApprover for AutoApprover {
    async fn approve(&self, _call: &DeferredToolCall) -> DeferredToolDecision {
        DeferredToolDecision::Approved
    }
}

/// Auto-deny all tool calls.
#[derive(Debug, Clone)]
pub struct AutoDenier {
    message: String,
}

impl AutoDenier {
    /// Create a new auto-denier with the given message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl ToolApprover for AutoDenier {
    async fn approve(&self, _call: &DeferredToolCall) -> DeferredToolDecision {
        DeferredToolDecision::Denied(self.message.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deferred_tool_call() {
        let call = DeferredToolCall::new("my_tool", serde_json::json!({"x": 1}))
            .with_tool_call_id("call_123");

        assert_eq!(call.tool_name, "my_tool");
        assert_eq!(call.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_deferred_tool_requests() {
        let mut requests = DeferredToolRequests::new();
        assert!(requests.is_empty());

        requests.add(DeferredToolCall::new("tool1", serde_json::json!({})));
        requests.add(DeferredToolCall::new("tool2", serde_json::json!({})));

        assert_eq!(requests.len(), 2);
        assert!(!requests.is_empty());
    }

    #[test]
    fn test_by_tool() {
        let mut requests = DeferredToolRequests::new();
        requests.add(DeferredToolCall::new("tool1", serde_json::json!({})));
        requests.add(DeferredToolCall::new("tool2", serde_json::json!({})));
        requests.add(DeferredToolCall::new("tool1", serde_json::json!({})));

        let tool1_calls = requests.by_tool("tool1");
        assert_eq!(tool1_calls.len(), 2);
    }

    #[test]
    fn test_approve_all() {
        let mut requests = DeferredToolRequests::new();
        requests.add(DeferredToolCall::new("tool1", serde_json::json!({})));
        requests.add(DeferredToolCall::new("tool2", serde_json::json!({})));

        let decisions = requests.approve_all();
        assert_eq!(decisions.len(), 2);
        assert!(decisions.all_approved());
    }

    #[test]
    fn test_deny_all() {
        let mut requests = DeferredToolRequests::new();
        requests.add(DeferredToolCall::new("tool1", serde_json::json!({})));

        let decisions = requests.deny_all("Not allowed");
        assert!(decisions.any_denied());
    }

    #[test]
    fn test_deferred_tool_decision() {
        let approved = DeferredToolDecision::Approved;
        assert!(approved.is_approved());
        assert!(!approved.is_denied());

        let denied = DeferredToolDecision::Denied("No".into());
        assert!(denied.is_denied());
        assert_eq!(denied.denial_message(), Some("No"));

        let custom = DeferredToolDecision::CustomResult(ToolReturn::text("custom"));
        assert!(custom.is_custom());
    }

    #[test]
    fn test_deferred_tool_result() {
        let result = DeferredToolResult::approved()
            .with_tool_call_id("id1");
        assert_eq!(result.tool_call_id, Some("id1".to_string()));

        let denied = DeferredToolResult::denied("Not allowed");
        assert!(denied.result.is_error());
    }

    #[test]
    fn test_deferred_tool_results() {
        let results = DeferredToolResults::approved(Some("id1".to_string()));
        assert_eq!(results.len(), 1);

        let denied = DeferredToolResults::denied(None, "Nope");
        assert_eq!(denied.len(), 1);
    }

    #[test]
    fn test_tool_denied() {
        let denied = ToolDenied::new("Not allowed");
        assert_eq!(denied.message, "Not allowed");
    }

    #[tokio::test]
    async fn test_auto_approver() {
        let approver = AutoApprover;
        let call = DeferredToolCall::new("test", serde_json::json!({}));
        let decision = approver.approve(&call).await;
        assert!(decision.is_approved());
    }

    #[tokio::test]
    async fn test_auto_denier() {
        let denier = AutoDenier::new("Denied");
        let call = DeferredToolCall::new("test", serde_json::json!({}));
        let decision = denier.approve(&call).await;
        assert!(decision.is_denied());
    }

    #[test]
    fn test_serde_roundtrip() {
        let call = DeferredToolCall::new("test", serde_json::json!({"x": 1}))
            .with_tool_call_id("id");
        let json = serde_json::to_string(&call).unwrap();
        let parsed: DeferredToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(call.tool_name, parsed.tool_name);
        assert_eq!(call.tool_call_id, parsed.tool_call_id);
    }
}
