# serdes-ai-streaming

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-streaming.svg)](https://crates.io/crates/serdes-ai-streaming)
[![Documentation](https://docs.rs/serdes-ai-streaming/badge.svg)](https://docs.rs/serdes-ai-streaming)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Streaming support for serdes-ai (SSE, WebSocket, async streams)

This crate provides streaming capabilities for SerdesAI:

- `AgentStreamEvent` enum for stream events
- SSE (Server-Sent Events) support
- WebSocket streaming
- Async stream utilities with backpressure

## Installation

```toml
[dependencies]
serdes-ai-streaming = "0.1"
```

## Usage

```rust
use serdes_ai_streaming::AgentStreamEvent;
use futures::StreamExt;

let mut stream = agent.run_stream("Write a poem", ()).await?;

while let Some(event) = stream.next().await {
    match event {
        AgentStreamEvent::TextDelta { content, .. } => {
            print!("{}", content);
        }
        AgentStreamEvent::End { .. } => break,
        _ => {}
    }
}
```

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.
