//! Response parsing utilities.

use serde::de::DeserializeOwned;
use serdes_ai_core::{Error, Result};

/// Parse a JSON response body.
pub fn parse_response<T: DeserializeOwned>(body: &str) -> Result<T> {
    serde_json::from_str(body).map_err(|e| Error::SerializationError(e))
}
