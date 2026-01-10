//! Cohere embedding model implementation.

use crate::embedding::Embedding;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::model::{EmbedInput, EmbeddingModel, EmbeddingOutput, EmbeddingSettings, InputType};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Cohere embedding model.
#[derive(Clone)]
pub struct CohereEmbeddingModel {
    model_name: String,
    client: Client,
    api_key: String,
    base_url: String,
    default_dimensions: usize,
}

impl CohereEmbeddingModel {
    /// Create a new Cohere embedding model.
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let name = model_name.into();
        let dimensions = Self::model_dimensions(&name);

        Self {
            model_name: name,
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.cohere.ai/v1".to_string(),
            default_dimensions: dimensions,
        }
    }

    /// Create from environment variable.
    pub fn from_env(model_name: impl Into<String>) -> EmbeddingResult<Self> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| EmbeddingError::config("COHERE_API_KEY not set"))?;
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
            "embed-english-v3.0" => 1024,
            "embed-multilingual-v3.0" => 1024,
            "embed-english-light-v3.0" => 384,
            "embed-multilingual-light-v3.0" => 384,
            "embed-english-v2.0" => 4096,
            "embed-multilingual-v2.0" => 768,
            _ => 1024,
        }
    }

    fn convert_input_type(input_type: Option<InputType>) -> &'static str {
        match input_type {
            Some(InputType::SearchQuery) => "search_query",
            Some(InputType::SearchDocument) => "search_document",
            Some(InputType::Classification) => "classification",
            Some(InputType::Clustering) => "clustering",
            None => "search_document",
        }
    }
}

impl std::fmt::Debug for CohereEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereEmbeddingModel")
            .field("model_name", &self.model_name)
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[derive(Debug, Serialize)]
struct CohereEmbedRequest {
    model: String,
    texts: Vec<String>,
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    embeddings: Vec<Vec<f32>>,
    meta: Option<CohereMeta>,
}

#[derive(Debug, Deserialize)]
struct CohereMeta {
    billed_units: Option<CohereBilledUnits>,
}

#[derive(Debug, Deserialize)]
struct CohereBilledUnits {
    input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CohereErrorResponse {
    message: String,
}

#[async_trait]
impl EmbeddingModel for CohereEmbeddingModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn dimensions(&self) -> usize {
        self.default_dimensions
    }

    fn max_tokens(&self) -> usize {
        512 // Cohere's default max
    }

    async fn embed(
        &self,
        input: EmbedInput,
        settings: &EmbeddingSettings,
    ) -> EmbeddingResult<EmbeddingOutput> {
        let texts = input.into_texts();
        let input_type = Self::convert_input_type(settings.input_type);

        let truncate = match settings.truncation {
            Some(crate::model::TruncationMode::End) => Some("END".to_string()),
            Some(crate::model::TruncationMode::Start) => Some("START".to_string()),
            Some(crate::model::TruncationMode::None) => Some("NONE".to_string()),
            None => None,
        };

        let request = CohereEmbedRequest {
            model: self.model_name.clone(),
            texts: texts.clone(),
            input_type: input_type.to_string(),
            truncate,
        };

        let response = self
            .client
            .post(format!("{}/embed", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();

            if let Ok(error_resp) = serde_json::from_str::<CohereErrorResponse>(&body) {
                if status.as_u16() == 429 {
                    return Err(EmbeddingError::RateLimited { retry_after: None });
                }
                return Err(EmbeddingError::Api(error_resp.message));
            }

            return Err(EmbeddingError::Http {
                status: status.as_u16(),
                body,
            });
        }

        let resp: CohereEmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Api(e.to_string()))?;

        let tokens = resp
            .meta
            .and_then(|m| m.billed_units)
            .and_then(|b| b.input_tokens);

        let embeddings: Vec<Embedding> = resp
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(i, vector)| {
                let text = texts.get(i).cloned();
                let mut emb = Embedding::new(vector)
                    .with_model(&self.model_name)
                    .with_index(i);
                if let Some(t) = text {
                    emb = emb.with_text(t);
                }
                emb
            })
            .collect();

        let mut output = EmbeddingOutput::new(embeddings, &self.model_name);
        if let Some(t) = tokens {
            output = output.with_tokens(t);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dimensions() {
        assert_eq!(
            CohereEmbeddingModel::model_dimensions("embed-english-v3.0"),
            1024
        );
        assert_eq!(
            CohereEmbeddingModel::model_dimensions("embed-english-light-v3.0"),
            384
        );
    }

    #[test]
    fn test_model_creation() {
        let model = CohereEmbeddingModel::new("embed-english-v3.0", "test-key");
        assert_eq!(model.name(), "embed-english-v3.0");
        assert_eq!(model.dimensions(), 1024);
    }

    #[test]
    fn test_convert_input_type() {
        assert_eq!(
            CohereEmbeddingModel::convert_input_type(Some(InputType::SearchQuery)),
            "search_query"
        );
        assert_eq!(
            CohereEmbeddingModel::convert_input_type(Some(InputType::SearchDocument)),
            "search_document"
        );
        assert_eq!(
            CohereEmbeddingModel::convert_input_type(None),
            "search_document"
        );
    }
}
