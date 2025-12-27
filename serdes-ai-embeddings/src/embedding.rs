//! Embedding types and batch operations.

use serde::{Deserialize, Serialize};

/// A vector embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Original text (if available).
    pub text: Option<String>,
    /// Model used to generate.
    pub model: Option<String>,
    /// Index in batch.
    pub index: Option<usize>,
}

impl Embedding {
    /// Create a new embedding.
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            text: None,
            model: None,
            index: None,
        }
    }

    /// Create from a slice.
    pub fn from_slice(vector: &[f32]) -> Self {
        Self::new(vector.to_vec())
    }

    /// Set the text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the index.
    pub fn with_index(mut self, index: usize) -> Self {
        self.index = Some(index);
        self
    }

    /// Get the dimensionality.
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }

    /// Get the vector as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    /// Calculate cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        crate::similarity::cosine_similarity(&self.vector, &other.vector)
    }

    /// Calculate dot product with another embedding.
    pub fn dot_product(&self, other: &Embedding) -> f32 {
        crate::similarity::dot_product(&self.vector, &other.vector)
    }

    /// Calculate Euclidean distance to another embedding.
    pub fn euclidean_distance(&self, other: &Embedding) -> f32 {
        crate::similarity::euclidean_distance(&self.vector, &other.vector)
    }

    /// Normalize the embedding vector in place.
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.vector == other.vector
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(vector: Vec<f32>) -> Self {
        Self::new(vector)
    }
}

impl From<&[f32]> for Embedding {
    fn from(vector: &[f32]) -> Self {
        Self::from_slice(vector)
    }
}

/// Batch of embeddings.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingBatch {
    /// Individual embeddings.
    pub embeddings: Vec<Embedding>,
    /// Total token usage.
    pub total_tokens: Option<u64>,
    /// Model used.
    pub model: Option<String>,
}

impl EmbeddingBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a vector of embeddings.
    pub fn from_vec(embeddings: Vec<Embedding>) -> Self {
        Self {
            embeddings,
            total_tokens: None,
            model: None,
        }
    }

    /// Set total tokens.
    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.total_tokens = Some(tokens);
        self
    }

    /// Set model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add an embedding.
    pub fn push(&mut self, embedding: Embedding) {
        self.embeddings.push(embedding);
    }

    /// Get the number of embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get embedding by index.
    pub fn get(&self, index: usize) -> Option<&Embedding> {
        self.embeddings.get(index)
    }

    /// Iterate over embeddings.
    pub fn iter(&self) -> impl Iterator<Item = &Embedding> {
        self.embeddings.iter()
    }

    /// Find most similar embedding to query.
    pub fn most_similar(&self, query: &Embedding) -> Option<(&Embedding, f32)> {
        self.embeddings
            .iter()
            .map(|e| (e, e.cosine_similarity(query)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find top K most similar embeddings.
    pub fn top_k(&self, query: &Embedding, k: usize) -> Vec<(&Embedding, f32)> {
        let mut scored: Vec<_> = self
            .embeddings
            .iter()
            .map(|e| (e, e.cosine_similarity(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Calculate pairwise similarity matrix.
    pub fn similarity_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.embeddings.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let sim = self.embeddings[i].cosine_similarity(&self.embeddings[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }
        matrix
    }
}

impl IntoIterator for EmbeddingBatch {
    type Item = Embedding;
    type IntoIter = std::vec::IntoIter<Embedding>;

    fn into_iter(self) -> Self::IntoIter {
        self.embeddings.into_iter()
    }
}

impl std::ops::Index<usize> for EmbeddingBatch {
    type Output = Embedding;

    fn index(&self, index: usize) -> &Self::Output {
        &self.embeddings[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_new() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dimensions(), 3);
    }

    #[test]
    fn test_embedding_with_metadata() {
        let emb = Embedding::new(vec![1.0, 2.0])
            .with_text("hello")
            .with_model("test-model")
            .with_index(0);

        assert_eq!(emb.text, Some("hello".to_string()));
        assert_eq!(emb.model, Some("test-model".to_string()));
        assert_eq!(emb.index, Some(0));
    }

    #[test]
    fn test_embedding_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0]);
        emb.normalize();

        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_similarity() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);
        let c = Embedding::new(vec![0.0, 1.0]);

        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);
        assert!((a.cosine_similarity(&c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_from_vec() {
        let batch = EmbeddingBatch::from_vec(vec![
            Embedding::new(vec![1.0, 0.0]),
            Embedding::new(vec![0.0, 1.0]),
        ]);

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_most_similar() {
        let batch = EmbeddingBatch::from_vec(vec![
            Embedding::new(vec![1.0, 0.0]),
            Embedding::new(vec![0.9, 0.1]),
            Embedding::new(vec![0.0, 1.0]),
        ]);

        let query = Embedding::new(vec![1.0, 0.0]);
        let (most, score) = batch.most_similar(&query).unwrap();

        assert_eq!(most.vector, vec![1.0, 0.0]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_top_k() {
        let batch = EmbeddingBatch::from_vec(vec![
            Embedding::new(vec![1.0, 0.0]),
            Embedding::new(vec![0.9, 0.1]),
            Embedding::new(vec![0.0, 1.0]),
        ]);

        let query = Embedding::new(vec![1.0, 0.0]);
        let top = batch.top_k(&query, 2);

        assert_eq!(top.len(), 2);
        assert!(top[0].1 >= top[1].1); // Sorted by score descending
    }

    #[test]
    fn test_batch_similarity_matrix() {
        let batch = EmbeddingBatch::from_vec(vec![
            Embedding::new(vec![1.0, 0.0]),
            Embedding::new(vec![0.0, 1.0]),
        ]);

        let matrix = batch.similarity_matrix();
        assert_eq!(matrix.len(), 2);
        assert!((matrix[0][0] - 1.0).abs() < 1e-6); // Self-similarity
        assert!((matrix[0][1] - 0.0).abs() < 1e-6); // Orthogonal
    }
}
