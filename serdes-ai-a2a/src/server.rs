//! HTTP server for the A2A protocol.
//!
//! This module provides an Axum-based HTTP server for exposing A2A endpoints.
//! It is only available when the `server` feature is enabled.

use crate::broker::Broker;
use crate::schema::{AgentCard, Message, TaskSendParams};
use crate::storage::Storage;
use crate::task::{Task, TaskStatus};
use crate::A2AServer;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Error response from the A2A server.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: None,
        }
    }

    pub fn with_code(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: Some(code.into()),
        }
    }
}

/// Task submission response.
#[derive(Debug, Serialize, Deserialize)]
pub struct TaskSubmitResponse {
    pub task_id: String,
    pub status: TaskStatus,
}

/// Task status response.
#[derive(Debug, Serialize, Deserialize)]
pub struct TaskStatusResponse {
    pub task_id: String,
    pub status: TaskStatus,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Shared state for the A2A HTTP handlers.
pub struct A2AState {
    pub agent_card: AgentCard,
    pub storage: Arc<dyn Storage>,
    pub broker: Arc<dyn Broker>,
}

impl<Deps, Output> A2AServer<Deps, Output>
where
    Deps: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Create an Axum router for the A2A endpoints.
    pub fn router(&self) -> Router {
        let state = Arc::new(A2AState {
            agent_card: self.agent_card(),
            storage: self.storage_arc(),
            broker: self.broker_arc(),
        });

        Router::new()
            .route("/.well-known/agent.json", get(get_agent_card))
            .route("/agent/card", get(get_agent_card))
            .route("/tasks/send", post(submit_task))
            .route("/tasks/:task_id", get(get_task_status))
            .route("/tasks/:task_id/cancel", post(cancel_task))
            .route("/health", get(health_check))
            .with_state(state)
    }

    /// Start serving on the given address.
    ///
    /// This will start an HTTP server and block until it's shut down.
    pub async fn serve(self, addr: impl Into<SocketAddr>) -> Result<(), ServerError> {
        let addr = addr.into();
        let router = self.router();

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Bind(e.to_string()))?;

        axum::serve(listener, router)
            .await
            .map_err(|e| ServerError::Serve(e.to_string()))?;

        Ok(())
    }
}

/// Server error types.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Failed to bind to address: {0}")]
    Bind(String),
    #[error("Server error: {0}")]
    Serve(String),
}

// Handler implementations

/// GET /.well-known/agent.json - Get agent card
async fn get_agent_card(State(state): State<Arc<A2AState>>) -> Json<AgentCard> {
    Json(state.agent_card.clone())
}

/// POST /tasks/send - Submit a new task
async fn submit_task(
    State(state): State<Arc<A2AState>>,
    Json(params): Json<TaskSendParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Create a new task
    let task = Task::new(params.thread_id, params.message);
    let task_id = task.id.clone();
    let status = task.status;

    // Save to storage
    if let Err(e) = state.storage.save_task(&task).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::with_code(
                format!("Failed to save task: {}", e),
                "storage_error",
            )),
        ));
    }

    // Submit to broker
    if let Err(e) = state.broker.submit_task(task).await {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse::with_code(
                format!("Failed to submit task: {}", e),
                "broker_error",
            )),
        ));
    }

    Ok((
        StatusCode::ACCEPTED,
        Json(TaskSubmitResponse { task_id, status }),
    ))
}

/// GET /tasks/:task_id - Get task status
async fn get_task_status(
    State(state): State<Arc<A2AState>>,
    Path(task_id): Path<String>,
) -> Result<Json<TaskStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.storage.get_task(&task_id).await {
        Ok(Some(task)) => Ok(Json(TaskStatusResponse {
            task_id: task.id,
            status: task.status,
            messages: task.messages,
            error: task.error,
        })),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::with_code(
                format!("Task not found: {}", task_id),
                "not_found",
            )),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::with_code(
                format!("Failed to get task: {}", e),
                "storage_error",
            )),
        )),
    }
}

/// POST /tasks/:task_id/cancel - Cancel a task
async fn cancel_task(
    State(state): State<Arc<A2AState>>,
    Path(task_id): Path<String>,
) -> Result<Json<TaskStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.storage.get_task(&task_id).await {
        Ok(Some(mut task)) => {
            if task.is_completed() {
                return Err((
                    StatusCode::CONFLICT,
                    Json(ErrorResponse::with_code(
                        "Task is already completed",
                        "already_completed",
                    )),
                ));
            }

            // Cancel in broker first (removes from queue and/or signals running task)
            state.broker.cancel_task(&task_id).await;

            // Then update task status in storage
            if let Err(e) = task.cancel() {
                return Err((
                    StatusCode::CONFLICT,
                    Json(ErrorResponse::with_code(
                        format!("Failed to cancel task: {}", e),
                        "invalid_state",
                    )),
                ));
            }

            if let Err(e) = state.storage.update_task(&task).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::with_code(
                        format!("Failed to cancel task: {}", e),
                        "storage_error",
                    )),
                ));
            }

            Ok(Json(TaskStatusResponse {
                task_id: task.id,
                status: task.status,
                messages: task.messages,
                error: task.error,
            }))
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::with_code(
                format!("Task not found: {}", task_id),
                "not_found",
            )),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::with_code(
                format!("Failed to get task: {}", e),
                "storage_error",
            )),
        )),
    }
}

/// GET /health - Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "a2a"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_response() {
        let error = ErrorResponse::new("Something went wrong");
        assert_eq!(error.error, "Something went wrong");
        assert!(error.code.is_none());
    }

    #[test]
    fn test_error_response_with_code() {
        let error = ErrorResponse::with_code("Not found", "not_found");
        assert_eq!(error.error, "Not found");
        assert_eq!(error.code, Some("not_found".to_string()));
    }
}
