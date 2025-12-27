//! HuggingFace Text Generation Inference (TGI) API types.

use serde::{Deserialize, Serialize};

/// Request to the HuggingFace Inference API.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateRequest {
    /// Input prompt text.
    pub inputs: String,
    /// Generation parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,
    /// Stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl GenerateRequest {
    /// Create a new generate request.
    pub fn new(inputs: impl Into<String>) -> Self {
        Self {
            inputs: inputs.into(),
            parameters: None,
            stream: None,
        }
    }

    /// Set generation parameters.
    #[must_use]
    pub fn with_parameters(mut self, params: GenerateParameters) -> Self {
        self.parameters = Some(params);
        self
    }

    /// Enable streaming.
    #[must_use]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}

/// Generation parameters for TGI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateParameters {
    /// Maximum number of new tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Repetition penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Whether to sample or use greedy decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Return full text (including prompt) or just generated text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_full_text: Option<bool>,
    /// Truncate inputs to fit within model's max length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<u32>,
    /// Return generation details (tokens, probabilities).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<bool>,
}

impl GenerateParameters {
    /// Create new parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max new tokens.
    #[must_use]
    pub fn max_new_tokens(mut self, n: u32) -> Self {
        self.max_new_tokens = Some(n);
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        // Enable sampling if temperature > 0
        if t > 0.0 {
            self.do_sample = Some(true);
        }
        self
    }

    /// Set top-p.
    #[must_use]
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set stop sequences.
    #[must_use]
    pub fn stop(mut self, sequences: Vec<String>) -> Self {
        self.stop = Some(sequences);
        self
    }
}

/// Response from the HuggingFace Inference API.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum GenerateResponse {
    /// Single generation result.
    Single(GenerationResult),
    /// Multiple generation results (batch).
    Batch(Vec<GenerationResult>),
}

impl GenerateResponse {
    /// Get the generated text from the first result.
    pub fn text(&self) -> Option<&str> {
        match self {
            GenerateResponse::Single(r) => Some(&r.generated_text),
            GenerateResponse::Batch(results) => results.first().map(|r| r.generated_text.as_str()),
        }
    }
}

/// Single generation result.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationResult {
    /// Generated text.
    pub generated_text: String,
    /// Generation details (if requested).
    #[serde(default)]
    pub details: Option<GenerationDetails>,
}

/// Generation details.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationDetails {
    /// Finish reason.
    pub finish_reason: Option<String>,
    /// Number of generated tokens.
    pub generated_tokens: Option<u32>,
    /// Seed used for generation.
    pub seed: Option<u64>,
    /// Prefill tokens info.
    pub prefill: Option<Vec<TokenInfo>>,
    /// Generated tokens info.
    pub tokens: Option<Vec<TokenInfo>>,
}

/// Token information.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenInfo {
    /// Token ID.
    pub id: u32,
    /// Token text.
    pub text: String,
    /// Log probability.
    pub logprob: Option<f32>,
    /// Whether this is a special token.
    pub special: Option<bool>,
}

/// Streaming response token.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamResponse {
    /// Token information.
    pub token: Option<StreamToken>,
    /// Generated text so far (on final chunk).
    pub generated_text: Option<String>,
    /// Generation details (on final chunk).
    pub details: Option<GenerationDetails>,
}

/// Streaming token.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamToken {
    /// Token ID.
    pub id: u32,
    /// Token text.
    pub text: String,
    /// Log probability.
    pub logprob: Option<f32>,
    /// Whether this is a special token.
    pub special: bool,
}

/// Error response from HuggingFace API.
#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceError {
    /// Error message.
    pub error: String,
    /// Error type.
    #[serde(rename = "error_type")]
    pub error_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_serialization() {
        let req = GenerateRequest::new("Hello, world!")
            .with_parameters(GenerateParameters::new().max_new_tokens(100).temperature(0.7));

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"inputs\":\"Hello, world!\""));
        assert!(json.contains("\"max_new_tokens\":100"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_generate_response_parsing() {
        let json = r#"{"generated_text": "Hello!"}
"#;
        let resp: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text(), Some("Hello!"));
    }

    #[test]
    fn test_batch_response_parsing() {
        let json = r#"[{"generated_text": "First"}, {"generated_text": "Second"}]"#;
        let resp: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text(), Some("First"));
    }
}
