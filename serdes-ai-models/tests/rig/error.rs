//! Error types for the Open Responses test rig.
//!
//! Errors are surfaced in two envelope shapes:
//!
//! - HTTP responses use the OpenAI-style body `{"error": {"code", "message"}}`
//!   with an appropriate status code.
//! - WebSocket frames use the event envelope
//!   `{"type": "error", "status_code", "error": {"code", "message"}}`,
//!   matching what the codex CLI parses (`codex-rs/codex-api` websocket
//!   client and the Open Responses websocket specification).
//!
//! The envelope types themselves live in the wire model
//! (`serdes_ai_models::openai::responses::wire`); this module defines the
//! rig's error enum and maps it onto those envelopes. Because the envelope
//! types are foreign, the original inherent `from_error` constructors are
//! expressed as the [`FromResponsesError`] extension trait so call sites
//! keep the `Envelope::from_error(&err)` form.

pub use serdes_ai_models::openai::responses::wire::codes;
pub use serdes_ai_models::openai::responses::wire::{
    ErrorBody, HttpErrorEnvelope, WsErrorEnvelope,
};

/// Errors produced while handling a Responses API turn.
#[derive(Debug, thiserror::Error)]
pub enum ResponsesError {
    /// The request is malformed or uses an unsupported feature.
    #[error("{0}")]
    InvalidRequest(String),
    /// A stored response with the given ID does not exist.
    #[error("{0}")]
    NotFound(String),
    /// `previous_response_id` is unknown to the store and, on websockets, to
    /// the connection-local session cache.
    #[error("previous response not found: {0}")]
    PreviousResponseNotFound(String),
    /// The backing model request failed.
    #[error("model error: {0}")]
    Model(String),
    /// The websocket connection outlived its allowed lifetime.
    #[error("websocket connection lifetime limit reached")]
    ConnectionLimitReached,
}

impl ResponsesError {
    /// HTTP status code for the error.
    #[must_use]
    pub fn status(&self) -> u16 {
        match self {
            Self::InvalidRequest(_) => 400,
            Self::NotFound(_) => 404,
            Self::PreviousResponseNotFound(_) => 404,
            Self::Model(_) => 502,
            Self::ConnectionLimitReached => 429,
        }
    }

    /// Stable error code for the error.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => codes::INVALID_REQUEST_ERROR,
            Self::NotFound(_) => codes::NOT_FOUND_ERROR,
            Self::PreviousResponseNotFound(_) => codes::PREVIOUS_RESPONSE_NOT_FOUND,
            Self::Model(_) => codes::MODEL_ERROR,
            Self::ConnectionLimitReached => codes::WEBSOCKET_CONNECTION_LIMIT_REACHED,
        }
    }

    /// The `previous_response_id` this error refers to, if any.
    ///
    /// Websocket sessions use this to evict a stale continuation ID after a
    /// failed chain so the client is pushed to replay full input.
    #[must_use]
    pub fn previous_response_id(&self) -> Option<String> {
        match self {
            Self::PreviousResponseNotFound(id) => Some(id.clone()),
            _ => None,
        }
    }

    /// The error body carried by both envelope shapes.
    #[must_use]
    pub fn body(&self) -> ErrorBody {
        ErrorBody {
            code: self.code().to_string(),
            message: self.to_string(),
            param: None,
        }
    }
}

/// Builds the wire error envelopes from a [`ResponsesError`].
pub trait FromResponsesError: Sized {
    /// Build an envelope from an error.
    fn from_error(err: &ResponsesError) -> Self;
}

impl FromResponsesError for HttpErrorEnvelope {
    fn from_error(err: &ResponsesError) -> Self {
        Self { error: err.body() }
    }
}

impl FromResponsesError for WsErrorEnvelope {
    fn from_error(err: &ResponsesError) -> Self {
        Self {
            kind: "error".to_string(),
            status_code: err.status(),
            error: err.body(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_and_codes() {
        assert_eq!(ResponsesError::InvalidRequest("x".into()).status(), 400);
        assert_eq!(
            ResponsesError::PreviousResponseNotFound("resp_x".into()).status(),
            404
        );
        assert_eq!(
            ResponsesError::PreviousResponseNotFound("resp_x".into()).code(),
            codes::PREVIOUS_RESPONSE_NOT_FOUND
        );
        assert_eq!(ResponsesError::ConnectionLimitReached.status(), 429);
        assert_eq!(
            ResponsesError::ConnectionLimitReached.code(),
            codes::WEBSOCKET_CONNECTION_LIMIT_REACHED
        );
    }

    #[test]
    fn ws_envelope_shape() {
        let err = ResponsesError::Model("boom".into());
        let json = WsErrorEnvelope::from_error(&err).to_json();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"status_code\":502"));
        assert!(json.contains("\"code\":\"model_error\""));
        assert!(json.contains("\"message\":\"model error: boom\""));
    }

    #[test]
    fn http_envelope_shape() {
        let err = ResponsesError::NotFound("no such response".into());
        let json = serde_json::to_string(&HttpErrorEnvelope::from_error(&err)).unwrap();
        assert_eq!(
            json,
            "{\"error\":{\"code\":\"not_found_error\",\"message\":\"no such response\"}}"
        );
    }
}
