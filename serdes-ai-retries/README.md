# serdes-ai-retries

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-retries.svg)](https://crates.io/crates/serdes-ai-retries)
[![Documentation](https://docs.rs/serdes-ai-retries/badge.svg)](https://docs.rs/serdes-ai-retries)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Retry strategies and error handling for serdes-ai

This crate provides retry capabilities for SerdesAI:

- Configurable retry strategies
- Exponential backoff with jitter
- Rate limit handling
- Transient error detection

## Installation

```toml
[dependencies]
serdes-ai-retries = "0.1"
```

## Usage

```rust
use serdes_ai_retries::{RetryStrategy, ExponentialBackoff};

let strategy = ExponentialBackoff::new()
    .max_retries(3)
    .initial_delay(Duration::from_millis(100))
    .max_delay(Duration::from_secs(10));

let agent = Agent::new(model)
    .retry_strategy(strategy)
    .build();
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.
