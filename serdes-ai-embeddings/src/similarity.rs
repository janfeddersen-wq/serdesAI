//! Similarity and distance functions for embeddings.

/// Calculate cosine similarity between two vectors.
///
/// Returns a value between -1 and 1, where:
/// - 1 means identical direction
/// - 0 means orthogonal
/// - -1 means opposite direction
///
/// # Examples
///
/// ```
/// use serdes_ai_embeddings::cosine_similarity;
///
/// let a = [1.0, 0.0];
/// let b = [1.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
///
/// let c = [0.0, 1.0];
/// assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Calculate dot product between two vectors.
///
/// # Examples
///
/// ```
/// use serdes_ai_embeddings::dot_product;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
/// ```
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate Euclidean distance between two vectors.
///
/// # Examples
///
/// ```
/// use serdes_ai_embeddings::euclidean_distance;
///
/// let a = [0.0, 0.0];
/// let b = [3.0, 4.0];
/// assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
/// ```
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan (L1) distance between two vectors.
///
/// # Examples
///
/// ```
/// use serdes_ai_embeddings::manhattan_distance;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// assert!((manhattan_distance(&a, &b) - 9.0).abs() < 1e-6);
/// ```
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Normalize a vector to unit length.
///
/// # Examples
///
/// ```
/// use serdes_ai_embeddings::normalize;
///
/// let v = vec![3.0, 4.0];
/// let n = normalize(&v);
/// let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-6);
/// ```
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Calculate the centroid of a set of vectors.
///
/// # Examples
///
/// ```
/// use serdes_ai_embeddings::centroid;
///
/// let vectors = vec![
///     vec![0.0, 0.0],
///     vec![2.0, 2.0],
/// ];
/// let c = centroid(&vectors);
/// assert!((c[0] - 1.0).abs() < 1e-6);
/// assert!((c[1] - 1.0).abs() < 1e-6);
/// ```
pub fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let n = vectors.len() as f32;
    let dims = vectors[0].len();
    let mut result = vec![0.0; dims];

    for v in vectors {
        for (i, val) in v.iter().enumerate() {
            result[i] += val / n;
        }
    }

    result
}

/// Convert cosine similarity to angular distance (0 to 1).
pub fn angular_distance(cosine_sim: f32) -> f32 {
    (1.0 - cosine_sim) / 2.0
}

/// Calculate pairwise cosine similarities.
///
/// Returns an NxM matrix where N is the number of query vectors
/// and M is the number of document vectors.
pub fn pairwise_cosine(queries: &[Vec<f32>], documents: &[Vec<f32>]) -> Vec<Vec<f32>> {
    queries
        .iter()
        .map(|q| documents.iter().map(|d| cosine_similarity(q, d)).collect())
        .collect()
}

/// Find top K most similar vectors.
pub fn top_k_similar(query: &[f32], candidates: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<_> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(query, c)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Calculate weighted average of embeddings.
pub fn weighted_average(embeddings: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    assert_eq!(
        embeddings.len(),
        weights.len(),
        "Number of embeddings must match weights"
    );

    if embeddings.is_empty() {
        return Vec::new();
    }

    let dims = embeddings[0].len();
    let total_weight: f32 = weights.iter().sum();
    let mut result = vec![0.0; dims];

    for (emb, &weight) in embeddings.iter().zip(weights.iter()) {
        let w = weight / total_weight;
        for (i, val) in emb.iter().enumerate() {
            result[i] += val * w;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0];
        let b = vec![4.0, 6.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn test_centroid() {
        let vectors = vec![vec![0.0, 0.0], vec![2.0, 2.0], vec![4.0, 4.0]];
        let c = centroid(&vectors);
        assert!((c[0] - 2.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_similar() {
        let query = vec![1.0, 0.0];
        let candidates = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];

        let top = top_k_similar(&query, &candidates, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // Most similar is first candidate
    }

    #[test]
    fn test_weighted_average() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let weights = vec![1.0, 1.0];
        let avg = weighted_average(&embeddings, &weights);
        assert!((avg[0] - 0.5).abs() < 1e-6);
        assert!((avg[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_cosine() {
        let queries = vec![vec![1.0, 0.0]];
        let docs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let matrix = pairwise_cosine(&queries, &docs);

        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 2);
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[0][1] - 0.0).abs() < 1e-6);
    }
}
