//! OpenAI embedding model implementation.

use crate::embedding::Embedding;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::model::{EmbedInput, EmbeddingModel, EmbeddingOutput, EmbeddingSettings};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// OpenAI embedding model.
#[derive(Clone)]
pub struct OpenAIEmbeddingModel {
    model_name: String,
    client: Client,
    api_key: String,
    base_url: String,
    default_dimensions: usize,
}

impl OpenAIEmbeddingModel {
    /// Create a new OpenAI embedding model.
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let name = model_name.into();
        let dimensions = Self::model_dimensions(&name);

        Self {
            model_name: name,
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            default_dimensions: dimensions,
        }
    }

    /// Create from environment variable.
    pub fn from_env(model_name: impl Into<String>) -> EmbeddingResult<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| EmbeddingError::config("OPENAI_API_KEY not set"))?;
        Ok(Self::new(model_name, api_key))
    }

    /// Set custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set custom HTTP client.
    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    fn model_dimensions(name: &str) -> usize {
        match name {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        }
    }

    fn model_max_tokens(name: &str) -> usize {
        match name {
            "text-embedding-3-small" => 8191,
            "text-embedding-3-large" => 8191,
            "text-embedding-ada-002" => 8191,
            _ => 8191,
        }
    }
}

impl std::fmt::Debug for OpenAIEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbeddingModel")
            .field("model_name", &self.model_name)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIUsage {
    prompt_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

#[async_trait]
impl EmbeddingModel for OpenAIEmbeddingModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn dimensions(&self) -> usize {
        self.default_dimensions
    }

    fn max_tokens(&self) -> usize {
        Self::model_max_tokens(&self.model_name)
    }

    async fn embed(
        &self,
        input: EmbedInput,
        settings: &EmbeddingSettings,
    ) -> EmbeddingResult<EmbeddingOutput> {
        let texts = input.into_texts();

        let request = OpenAIEmbeddingRequest {
            model: self.model_name.clone(),
            input: texts.clone(),
            dimensions: settings.dimensions,
            user: settings.user.clone(),
        };

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();

            // Try to parse error response
            if let Ok(error_resp) = serde_json::from_str::<OpenAIErrorResponse>(&body) {
                if status.as_u16() == 429 {
                    return Err(EmbeddingError::RateLimited { retry_after: None });
                }
                return Err(EmbeddingError::Api(error_resp.error.message));
            }

            return Err(EmbeddingError::Http {
                status: status.as_u16(),
                body,
            });
        }

        let resp: OpenAIEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        // Sort by index and create embeddings
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);

        let embeddings: Vec<Embedding> = data
            .into_iter()
            .map(|d| {
                let text = texts.get(d.index).cloned();
                let mut emb = Embedding::new(d.embedding)
                    .with_model(&resp.model)
                    .with_index(d.index);
                if let Some(t) = text {
                    emb = emb.with_text(t);
                }
                emb
            })
            .collect();

        Ok(EmbeddingOutput::new(embeddings, &resp.model).with_tokens(resp.usage.total_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(OpenAIEmbeddingModel::model_dimensions("text-embedding-3-small"), 1536);
        assert_eq!(OpenAIEmbeddingModel::model_dimensions("text-embedding-3-large"), 3072);
    }

    #[test]
    fn test_model_creation() {
        let model = OpenAIEmbeddingModel::new("text-embedding-3-small", "test-key");
        assert_eq!(model.name(), "text-embedding-3-small");
        assert_eq!(model.dimensions(), 1536);
    }

    #[test]
    fn test_with_base_url() {
        let model = OpenAIEmbeddingModel::new("test", "key")
            .with_base_url("https://custom.api.com");
        assert_eq!(model.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_debug() {
        let model = OpenAIEmbeddingModel::new("test", "secret-key");
        let debug = format!("{:?}", model);
        assert!(debug.contains("test"));
        assert!(!debug.contains("secret")); // Key should not be in debug
    }
}
