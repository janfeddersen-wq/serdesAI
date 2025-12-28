//! RAG (Retrieval-Augmented Generation) example.
//!
//! This example demonstrates how to use embeddings for semantic search
//! and augment LLM responses with retrieved context.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=your-key cargo run --example rag --features embeddings
//! ```

use serdes_ai::prelude::*;

/// A simple document store.
#[derive(Debug, Clone)]
struct Document {
    id: String,
    title: String,
    content: String,
    embedding: Option<Vec<f32>>,
}

/// A simple vector store for documents.
struct VectorStore {
    documents: Vec<Document>,
}

impl VectorStore {
    fn new() -> Self {
        Self { documents: Vec::new() }
    }

    fn add_document(&mut self, doc: Document) {
        self.documents.push(doc);
    }

    /// Search for similar documents using cosine similarity.
    fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(&Document, f32)> {
        let mut results: Vec<_> = self
            .documents
            .iter()
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| {
                    let similarity = cosine_similarity(query_embedding, emb);
                    (doc, similarity)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        results
    }
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Sample knowledge base about Rust.
fn create_knowledge_base() -> Vec<Document> {
    vec![
        Document {
            id: "1".to_string(),
            title: "Rust Memory Safety".to_string(),
            content: "Rust guarantees memory safety without garbage collection through its \
                     ownership system. Every value has a single owner, and the value is \
                     dropped when the owner goes out of scope. The borrow checker ensures \
                     references are always valid.".to_string(),
            embedding: None,
        },
        Document {
            id: "2".to_string(),
            title: "Rust Performance".to_string(),
            content: "Rust provides zero-cost abstractions, meaning high-level features \
                     compile down to efficient low-level code. There's no runtime or \
                     garbage collector overhead. Rust code often matches or exceeds \
                     C/C++ performance while being memory safe.".to_string(),
            embedding: None,
        },
        Document {
            id: "3".to_string(),
            title: "Rust Concurrency".to_string(),
            content: "Rust's type system prevents data races at compile time. The Send \
                     and Sync traits indicate which types can be transferred or shared \
                     between threads. This enables 'fearless concurrency' where the \
                     compiler catches threading bugs before runtime.".to_string(),
            embedding: None,
        },
        Document {
            id: "4".to_string(),
            title: "Cargo Package Manager".to_string(),
            content: "Cargo is Rust's package manager and build system. It handles \
                     dependency management, building, testing, and documentation. \
                     The crates.io registry hosts over 100,000 packages. Cargo.toml \
                     defines project metadata and dependencies.".to_string(),
            embedding: None,
        },
        Document {
            id: "5".to_string(),
            title: "Rust Error Handling".to_string(),
            content: "Rust uses Result and Option types for error handling instead of \
                     exceptions. The ? operator provides ergonomic error propagation. \
                     This makes error handling explicit and prevents silent failures. \
                     Panics are used for unrecoverable errors.".to_string(),
            embedding: None,
        },
    ]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("📚 RAG (Retrieval-Augmented Generation) Example\n");

    // For this example, we'll simulate embeddings
    // In production, you'd use: serdes_ai::embeddings::OpenAIEmbeddings
    println!("1️⃣  Loading knowledge base...");
    let mut store = VectorStore::new();
    
    for mut doc in create_knowledge_base() {
        // Simulate embedding (in production, call embedding API)
        doc.embedding = Some(simulate_embedding(&doc.content));
        println!("   ✅ Embedded: {}", doc.title);
        store.add_document(doc);
    }

    // User queries
    let queries = [
        "How does Rust handle memory without garbage collection?",
        "What makes Rust good for concurrent programming?",
        "How do I manage dependencies in Rust?",
    ];

    // Create the RAG agent
    let agent = Agent::builder()
        .model("openai:gpt-4o")
        .system_prompt(
            "You are a helpful Rust programming assistant. \
             Answer questions using the provided context. \
             If the context doesn't contain relevant information, \
             say so and provide general knowledge.",
        )
        .build()?;

    for query in queries {
        println!();
        println!("=".repeat(60));
        println!("💬 Query: {}\n", query);

        // Step 1: Embed the query
        let query_embedding = simulate_embedding(query);

        // Step 2: Retrieve relevant documents
        println!("2️⃣  Retrieving relevant context...");
        let results = store.search(&query_embedding, 2);
        
        let context: String = results
            .iter()
            .map(|(doc, score)| {
                println!("   📄 {} (score: {:.3})", doc.title, score);
                format!("### {}\n{}\n", doc.title, doc.content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Step 3: Generate response with context
        println!("\n3️⃣  Generating response...");
        
        let prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nAnswer based on the context above:",
            context, query
        );

        let response = agent.run(&prompt, ()).await?;

        println!("\n🤖 Answer:\n{}", response.output());
    }

    println!("\n");
    println!("=".repeat(60));
    println!("✅ RAG pipeline complete!");

    Ok(())
}

/// Simulate embedding generation.
/// In production, use actual embedding models.
fn simulate_embedding(text: &str) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    // Create a simple hash-based "embedding" for demo purposes
    // Real embeddings would be 1536-dimensional (OpenAI) or 768-dimensional
    let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
    let mut embedding = vec![0.0f32; 64];

    for (i, word) in words.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let hash = hasher.finish();
        
        for j in 0..64 {
            embedding[j] += ((hash >> (j % 64)) as f32 * 0.001) / (i as f32 + 1.0);
        }
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    }

    embedding
}
