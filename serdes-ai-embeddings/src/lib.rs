//! # serdes-ai-embeddings
//!
//! Embedding models and vector operations for serdes-ai.
//!
//! This crate provides infrastructure for generating embeddings from text
//! and performing vector similarity operations.
//!
//! ## Core Concepts
//!
//! - **[`EmbeddingModel`]**: Trait for embedding model implementations
//! - **[`Embedding`]**: Vector representation with metadata
//! - **[`Embedder`]**: High-level interface for embeddings
//! - **Similarity functions**: Cosine, dot product, Euclidean distance
//!
//! ## Feature Flags
//!
//! - `openai` (default): OpenAI embedding models
//! - `cohere`: Cohere embedding models
//! - `voyage`: Voyage AI embeddings
//! - `ollama`: Local Ollama embeddings
//! - `full`: All providers
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_embeddings::{Embedder, EmbeddingModel};
//!
//! // Using the high-level interface
//! let embedder = Embedder::from_env("openai:text-embedding-3-small")?;
//!
//! // Single embedding
//! let result = embedder.embed_query("Hello, world!").await?;
//! let embedding = result.embedding().unwrap();
//!
//! // Batch embeddings
//! let result = embedder.embed_documents(vec![
//!     "First document".into(),
//!     "Second document".into(),
//! ]).await?;
//!
//! // Similarity
//! let similarity = result.embeddings[0].cosine_similarity(&result.embeddings[1]);
//! ```
//!
//! ## Direct Model Usage
//!
//! ```ignore
//! use serdes_ai_embeddings::OpenAIEmbeddingModel;
//!
//! let model = OpenAIEmbeddingModel::from_env("text-embedding-3-small")?;
//! let result = model.embed_query("Hello!").await?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod embedder;
pub mod embedding;
pub mod error;
pub mod model;
pub mod similarity;

#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub mod openai;

#[cfg(feature = "cohere")]
#[cfg_attr(docsrs, doc(cfg(feature = "cohere")))]
pub mod cohere;

// Re-exports
pub use embedder::{infer_embedding_model, Embedder, EmbedderBuilder};
pub use embedding::{Embedding, EmbeddingBatch};
pub use error::{EmbeddingError, EmbeddingResult};
pub use model::{
    BoxedEmbeddingModel, EmbedInput, EmbeddingModel, EmbeddingOutput, EmbeddingSettings,
    EncodingFormat, InputType, TruncationMode,
};
pub use similarity::{
    angular_distance, centroid, cosine_similarity, dot_product, euclidean_distance,
    manhattan_distance, normalize, pairwise_cosine, top_k_similar, weighted_average,
};

#[cfg(feature = "openai")]
pub use openai::OpenAIEmbeddingModel;

#[cfg(feature = "cohere")]
pub use cohere::CohereEmbeddingModel;

/// Prelude for common imports.
pub mod prelude {
    pub use crate::{
        cosine_similarity, dot_product, Embedder, Embedding, EmbeddingBatch, EmbeddingError,
        EmbeddingModel, EmbeddingOutput, EmbeddingResult,
    };

    #[cfg(feature = "openai")]
    pub use crate::OpenAIEmbeddingModel;

    #[cfg(feature = "cohere")]
    pub use crate::CohereEmbeddingModel;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dimensions(), 3);
    }

    #[test]
    fn test_similarity_functions() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }
}
