# serdes-ai-providers

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-providers.svg)](https://crates.io/crates/serdes-ai-providers)
[![Documentation](https://docs.rs/serdes-ai-providers/badge.svg)](https://docs.rs/serdes-ai-providers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Provider abstractions and registry for serdes-ai

This crate provides the provider abstraction layer for SerdesAI, enabling:

- Provider trait definitions
- Provider registry for dynamic model selection
- Common provider utilities
- Authentication and configuration helpers

## Installation

```toml
[dependencies]
serdes-ai-providers = "0.1"
```

## Usage

```rust
use serdes_ai_providers::{Provider, ProviderRegistry};

let registry = ProviderRegistry::default();
let provider = registry.get("openai")?;
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.
