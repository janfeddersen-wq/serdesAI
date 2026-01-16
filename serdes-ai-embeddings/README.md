# serdes-ai-embeddings

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-embeddings.svg)](https://crates.io/crates/serdes-ai-embeddings)
[![Documentation](https://docs.rs/serdes-ai-embeddings/badge.svg)](https://docs.rs/serdes-ai-embeddings)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Embedding models and vector operations for serdes-ai

This crate provides embedding support for SerdesAI:

- `EmbeddingModel` trait
- OpenAI, Cohere, and other embedding providers
- Vector similarity operations
- Batch embedding utilities

## Installation

```toml
[dependencies]
serdes-ai-embeddings = "0.1"
```

## Usage

```rust
use serdes_ai_embeddings::{EmbeddingModel, OpenAIEmbeddings};

let model = OpenAIEmbeddings::from_env("text-embedding-3-small")?;
let embeddings = model.embed(&["Hello, world!", "Goodbye!"]).await?;

// Calculate similarity
let similarity = embeddings[0].cosine_similarity(&embeddings[1]);
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.
