//! Embedding model trait and types.

use crate::embedding::Embedding;
use crate::error::{EmbeddingError, EmbeddingResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Result of an embedding operation.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// The embedding vectors.
    pub embeddings: Vec<Embedding>,
    /// Total tokens used.
    pub total_tokens: Option<u64>,
    /// Model name used.
    pub model: String,
}

impl EmbeddingOutput {
    /// Create a new embedding output.
    pub fn new(embeddings: Vec<Embedding>, model: impl Into<String>) -> Self {
        Self {
            embeddings,
            total_tokens: None,
            model: model.into(),
        }
    }

    /// Set the token count.
    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.total_tokens = Some(tokens);
        self
    }

    /// Get the first (single) embedding.
    pub fn embedding(&self) -> Option<&Embedding> {
        self.embeddings.first()
    }

    /// Get the dimensionality.
    pub fn dimensions(&self) -> Option<usize> {
        self.embeddings.first().map(|e| e.dimensions())
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get number of embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }
}

/// Input type for embedding operations.
#[derive(Debug, Clone)]
pub enum EmbedInput {
    /// Single text query.
    Query(String),
    /// Multiple documents.
    Documents(Vec<String>),
}

impl EmbedInput {
    /// Get the number of texts.
    pub fn len(&self) -> usize {
        match self {
            Self::Query(_) => 1,
            Self::Documents(docs) => docs.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Query(q) => q.is_empty(),
            Self::Documents(docs) => docs.is_empty(),
        }
    }

    /// Convert to list of strings.
    pub fn into_texts(self) -> Vec<String> {
        match self {
            Self::Query(q) => vec![q],
            Self::Documents(docs) => docs,
        }
    }

    /// Get as list of string references.
    pub fn texts(&self) -> Vec<&str> {
        match self {
            Self::Query(q) => vec![q.as_str()],
            Self::Documents(docs) => docs.iter().map(|s| s.as_str()).collect(),
        }
    }
}

impl From<&str> for EmbedInput {
    fn from(s: &str) -> Self {
        Self::Query(s.to_string())
    }
}

impl From<String> for EmbedInput {
    fn from(s: String) -> Self {
        Self::Query(s)
    }
}

impl From<Vec<String>> for EmbedInput {
    fn from(docs: Vec<String>) -> Self {
        Self::Documents(docs)
    }
}

impl From<Vec<&str>> for EmbedInput {
    fn from(docs: Vec<&str>) -> Self {
        Self::Documents(docs.into_iter().map(String::from).collect())
    }
}

/// Encoding format for embeddings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Standard floating point.
    #[default]
    Float,
    /// Base64 encoded.
    Base64,
}

/// Settings for embedding requests.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingSettings {
    /// Override dimensions (if model supports).
    pub dimensions: Option<usize>,
    /// Encoding format.
    pub encoding_format: Option<EncodingFormat>,
    /// User identifier for tracking.
    pub user: Option<String>,
    /// Input type hint (query vs document).
    pub input_type: Option<InputType>,
    /// Truncation mode.
    pub truncation: Option<TruncationMode>,
}

impl EmbeddingSettings {
    /// Create new default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set dimensions.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set encoding format.
    pub fn encoding_format(mut self, format: EncodingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Set user identifier.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set input type.
    pub fn input_type(mut self, input_type: InputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Set truncation mode.
    pub fn truncation(mut self, mode: TruncationMode) -> Self {
        self.truncation = Some(mode);
        self
    }
}

/// Type of input for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    /// Search query (optimized for retrieval).
    SearchQuery,
    /// Document to be indexed.
    SearchDocument,
    /// Classification input.
    Classification,
    /// Clustering input.
    Clustering,
}

/// Truncation mode for long inputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TruncationMode {
    /// Truncate to fit model's context.
    #[default]
    End,
    /// Truncate from start.
    Start,
    /// No truncation (error if too long).
    None,
}

/// Core trait for embedding models.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Get the model name.
    fn name(&self) -> &str;

    /// Get the default embedding dimensions.
    fn dimensions(&self) -> usize;

    /// Get the maximum tokens per input.
    fn max_tokens(&self) -> usize {
        8192 // Default, override for specific models
    }

    /// Generate embeddings.
    async fn embed(
        &self,
        input: EmbedInput,
        settings: &EmbeddingSettings,
    ) -> EmbeddingResult<EmbeddingOutput>;

    /// Embed a single query.
    async fn embed_query(&self, query: &str) -> EmbeddingResult<EmbeddingOutput> {
        self.embed(
            EmbedInput::Query(query.to_string()),
            &EmbeddingSettings::default().input_type(InputType::SearchQuery),
        )
        .await
    }

    /// Embed multiple documents.
    async fn embed_documents(&self, docs: Vec<String>) -> EmbeddingResult<EmbeddingOutput> {
        self.embed(
            EmbedInput::Documents(docs),
            &EmbeddingSettings::default().input_type(InputType::SearchDocument),
        )
        .await
    }

    /// Count tokens in text (if supported).
    async fn count_tokens(&self, _text: &str) -> EmbeddingResult<u64> {
        Err(EmbeddingError::NotSupported("Token counting".into()))
    }
}

/// Boxed embedding model for dynamic dispatch.
pub type BoxedEmbeddingModel = Box<dyn EmbeddingModel>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_input_from_str() {
        let input: EmbedInput = "hello".into();
        assert!(matches!(input, EmbedInput::Query(_)));
        assert_eq!(input.len(), 1);
    }

    #[test]
    fn test_embed_input_from_vec() {
        let input: EmbedInput = vec!["a", "b", "c"].into();
        assert!(matches!(input, EmbedInput::Documents(_)));
        assert_eq!(input.len(), 3);
    }

    #[test]
    fn test_embedding_output() {
        let embeddings = vec![
            Embedding::new(vec![1.0, 2.0, 3.0]),
            Embedding::new(vec![4.0, 5.0, 6.0]),
        ];
        let output = EmbeddingOutput::new(embeddings, "test-model").with_tokens(100);

        assert_eq!(output.len(), 2);
        assert_eq!(output.dimensions(), Some(3));
        assert_eq!(output.total_tokens, Some(100));
    }

    #[test]
    fn test_embedding_settings() {
        let settings = EmbeddingSettings::new()
            .dimensions(1536)
            .input_type(InputType::SearchQuery)
            .user("user-123");

        assert_eq!(settings.dimensions, Some(1536));
        assert_eq!(settings.input_type, Some(InputType::SearchQuery));
    }
}
