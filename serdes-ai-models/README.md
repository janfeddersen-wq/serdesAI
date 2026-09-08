# serdes-ai-models

[![Crates.io](https://img.shields.io/crates/v/serdes-ai-models.svg)](https://crates.io/crates/serdes-ai-models)
[![Documentation](https://docs.rs/serdes-ai-models/badge.svg)](https://docs.rs/serdes-ai-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE)

> Model trait and provider implementations for serdes-ai

This crate defines the `Model` trait and provides implementations for various LLM providers:

- OpenAI (GPT-4, GPT-4o, o1, o3)
- Anthropic (Claude 3.5, Claude 4)
- Google (Gemini 1.5, Gemini 2.0)
- Groq (Llama, Mixtral)
- Mistral
- Ollama (local models)
- Azure OpenAI
- AWS Bedrock

## Installation

```toml
[dependencies]
serdes-ai-models = "0.1"
```

## Usage

```rust,ignore
use serdes_ai_models::{Model, ModelRetryExt, OpenAIChatModel, RetryPolicy};
use std::time::Duration;

let model = OpenAIChatModel::from_env("gpt-4o")?.with_retries(
    RetryPolicy::for_model_requests()
        .max_attempts(3)
        .total_timeout(Some(Duration::from_secs(30))),
);

let response = model.request(&messages, &settings, &params).await?;
```

Retries are opt-in. `RetryPolicy::disabled()` performs exactly one attempt. The
policy retries the same model; `FallbackModel` remains responsible for selecting
a different model. Streaming requests may be retried only while acquiring the
stream and before the first caller-visible event. Once an event is returned, later
stream errors pass through without replaying or concatenating another response.

## Responses API (OpenAI and Open Responses)

`OpenAIResponsesModel` speaks the Responses API over two transports. HTTP
(`POST {base_url}/responses`) is the default. The WebSocket transport dials the
base URL verbatim as the responses endpoint and requires the `responses-ws`
feature:

```rust,ignore
use serdes_ai_models::model::{Model, ModelRequestParameters};
use serdes_ai_models::openai::OpenAIResponsesModel;
use serdes_ai_models::openai::responses::Transport;

let model = OpenAIResponsesModel::new("gpt-5.1", api_key)
    .with_base_url("wss://api.openai.com/v1/responses")
    .with_transport(Transport::WebSocket)
    .with_header("Authorization", format!("Bearer {api_key}"))
    .with_session_chaining(true);
```

Session chaining (`with_session_chaining(true)`) keeps per-conversation state on
both transports: the websocket holds one socket per conversation and sends only
each turn's new input items with `store: false`; HTTP persists every turn with
`store: true` plus `previous_response_id` and streams SSE. Stale continuations
replay the full input, and connection failures reconnect before any
caller-visible event, so partial output is never duplicated.

The codex backend smoke test, including the ChatGPT OAuth PKCE flow, lives in
`serdes-ai-providers/examples/codex_haiku.rs`.

## Streaming fallback boundary

`FallbackModel` can select another model for acquisition errors or retryable
errors yielded before the first stream event. It polls and buffers at most one
event while selecting an attempt. Every event counts as caller-visible exposure,
including terminal/provider metadata. Once an event is returned, later errors
are propagated from that model and fallback never replays or concatenates
another model's output. Dropping the initial request future closes the current
stream and prevents another fallback attempt.

Same-model retries remain a separate opt-in layer through `RetryingModel`.

## Model failure contract

`serdes_ai_core::ModelFailure` is the authoritative, serializable classification
used by direct model calls, same-model retries, fallback selection, and agent
wrappers. It preserves the semantic kind, HTTP status, provider code,
`Retry-After`, and available provider/model/attempt context. `ModelError` remains
the concrete source-bearing error and implements `ClassifyModelFailure`.

Migration guidance:

- Use `ClassifyModelFailure::model_failure()` instead of matching retryability in
  each crate.
- `ProviderErrorKind` remains a compatibility alias for `ModelFailureKind`.
- Core `ModelApiError` and `ModelHttpError` convert into `ModelError` while
  remaining the source of the converted error.
- `serdes-ai-providers::ProviderError` is limited to provider discovery and
  configuration; model-call failures use `ModelError`.
- `serdes-ai-retries::RetryableError` remains limited to the legacy standalone
  HTTP retry client. Tool, user, cancellation, and output-validation failures
  retain their distinct semantics.

## Part of SerdesAI

This crate is part of the [SerdesAI](https://github.com/janfeddersen-wq/serdesAI) workspace.

For most use cases, you should use the main `serdes-ai` crate which re-exports these types.

## License

MIT License - see [LICENSE](https://github.com/janfeddersen-wq/serdesAI/blob/main/LICENSE) for details.
