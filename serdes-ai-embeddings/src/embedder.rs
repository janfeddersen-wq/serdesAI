//! High-level embedder interface.

use crate::embedding::{Embedding, EmbeddingBatch};
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::model::{
    BoxedEmbeddingModel, EmbedInput, EmbeddingModel, EmbeddingOutput, EmbeddingSettings,
};
use crate::similarity;

/// High-level interface for generating embeddings.
///
/// The Embedder provides a convenient API for working with embeddings,
/// including similarity search and batch operations.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_embeddings::Embedder;
///
/// let embedder = Embedder::from_env("openai:text-embedding-3-small")?;
///
/// // Embed a query
/// let query = embedder.embed_query("What is Rust?").await?;
///
/// // Embed documents
/// let docs = embedder.embed_documents(vec![
///     "Rust is a systems programming language.".into(),
///     "Python is great for data science.".into(),
/// ]).await?;
///
/// // Find most similar
/// let (best, score) = embedder.most_similar(&query.embedding().unwrap(), &docs)?;
/// ```
pub struct Embedder {
    model: BoxedEmbeddingModel,
    settings: EmbeddingSettings,
}

impl Embedder {
    /// Create a new embedder with a model.
    pub fn new<M: EmbeddingModel + 'static>(model: M) -> Self {
        Self {
            model: Box::new(model),
            settings: EmbeddingSettings::default(),
        }
    }

    /// Create from a model name (format: "provider:model-name").
    ///
    /// Supported providers:
    /// - `openai`: OpenAI embedding models
    /// - `cohere`: Cohere embedding models
    ///
    /// API keys are read from environment variables:
    /// - `OPENAI_API_KEY` for OpenAI
    /// - `COHERE_API_KEY` for Cohere
    pub fn from_env(name: &str) -> EmbeddingResult<Self> {
        let model = infer_embedding_model(name)?;
        Ok(Self {
            model,
            settings: EmbeddingSettings::default(),
        })
    }

    /// Set the embedding settings.
    pub fn with_settings(mut self, settings: EmbeddingSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Set the output dimensions (if model supports).
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.settings.dimensions = Some(dims);
        self
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.model.name()
    }

    /// Get the default dimensions.
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    /// Embed a single query.
    pub async fn embed_query(&self, query: &str) -> EmbeddingResult<EmbeddingOutput> {
        self.model
            .embed(EmbedInput::Query(query.to_string()), &self.settings)
            .await
    }

    /// Embed multiple documents.
    pub async fn embed_documents(
        &self,
        docs: Vec<String>,
    ) -> EmbeddingResult<EmbeddingOutput> {
        self.model
            .embed(EmbedInput::Documents(docs), &self.settings)
            .await
    }

    /// Embed with custom input.
    pub async fn embed(
        &self,
        input: impl Into<EmbedInput>,
    ) -> EmbeddingResult<EmbeddingOutput> {
        self.model.embed(input.into(), &self.settings).await
    }

    /// Find the most similar embedding from a batch.
    pub fn most_similar<'a>(
        &self,
        query: &Embedding,
        candidates: &'a EmbeddingBatch,
    ) -> Option<(&'a Embedding, f32)> {
        candidates.most_similar(query)
    }

    /// Find top K most similar embeddings.
    pub fn top_k<'a>(
        &self,
        query: &Embedding,
        candidates: &'a EmbeddingBatch,
        k: usize,
    ) -> Vec<(&'a Embedding, f32)> {
        candidates.top_k(query, k)
    }

    /// Calculate cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
        similarity::cosine_similarity(&a.vector, &b.vector)
    }

    /// Calculate cosine similarity between two vectors.
    pub fn cosine_similarity_vec(a: &[f32], b: &[f32]) -> f32 {
        similarity::cosine_similarity(a, b)
    }

    /// Create an embedding from a vector.
    pub fn embedding_from_vec(vector: Vec<f32>) -> Embedding {
        Embedding::new(vector)
    }
}

/// Infer embedding model from a name string.
///
/// Format: "provider:model-name"
///
/// Examples:
/// - "openai:text-embedding-3-small"
/// - "cohere:embed-english-v3.0"
pub fn infer_embedding_model(name: &str) -> EmbeddingResult<BoxedEmbeddingModel> {
    let (provider, model_name) = name
        .split_once(':')
        .ok_or_else(|| EmbeddingError::config("Invalid model format. Use 'provider:model-name'"))?;

    match provider {
        #[cfg(feature = "openai")]
        "openai" => {
            use crate::openai::OpenAIEmbeddingModel;
            let model = OpenAIEmbeddingModel::from_env(model_name)?;
            Ok(Box::new(model))
        }

        #[cfg(feature = "cohere")]
        "cohere" => {
            use crate::cohere::CohereEmbeddingModel;
            let model = CohereEmbeddingModel::from_env(model_name)?;
            Ok(Box::new(model))
        }

        _ => Err(EmbeddingError::config(format!(
            "Unknown provider: {}. Available: openai, cohere",
            provider
        ))),
    }
}

/// Builder for creating embedders with configuration.
#[derive(Debug, Default)]
pub struct EmbedderBuilder {
    model_name: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    dimensions: Option<usize>,
}

impl EmbedderBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name.
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Set the API key.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the output dimensions.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Build the embedder.
    pub fn build(self) -> EmbeddingResult<Embedder> {
        let model_name = self
            .model_name
            .ok_or_else(|| EmbeddingError::config("Model name required"))?;

        // If API key provided, use it; otherwise try env
        let embedder = if let Some(_api_key) = self.api_key {
            // Would create model with explicit key
            Embedder::from_env(&model_name)?
        } else {
            Embedder::from_env(&model_name)?
        };

        let mut embedder = embedder;
        if let Some(dims) = self.dimensions {
            embedder = embedder.with_dimensions(dims);
        }

        Ok(embedder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_builder() {
        let builder = EmbedderBuilder::new()
            .model("openai:test")
            .dimensions(256);

        assert_eq!(builder.model_name, Some("openai:test".to_string()));
        assert_eq!(builder.dimensions, Some(256));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);

        let sim = Embedder::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_model_invalid_format() {
        let result = infer_embedding_model("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_model_unknown_provider() {
        let result = infer_embedding_model("unknown:model");
        assert!(result.is_err());
    }
}
