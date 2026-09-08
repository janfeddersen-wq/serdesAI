# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-24

Combined release integrating PRs #51, #52, #53, #54 and #55. Streaming is now a
fully observable path: providers emit a terminal `StreamComplete` carrying token
usage, and the agent surfaces that usage through `AgentStreamEvent`.

### Added
- Every provider streaming path now ends with exactly one terminal `StreamComplete` event carrying the finish reason and token usage, so callers read final token counts from the stream itself instead of issuing a separate non-streaming request (#54):
  - OpenAI chat streaming buffers the `include_usage` chunk and emits the terminal event at `data: [DONE]`; OpenRouter, Azure and Groq inherit the parser.
  - Google and Antigravity close open parts with `PartEnd`; the terminal event carries the final chunk's `finishReason` and `usageMetadata`, mapping `cachedContentTokenCount` to `cache_read_tokens` (that wire reports no cache-creation count).
  - Cohere maps `stream-end` token counts; HuggingFace and ChatGPT OAuth map `response.completed` usage; the Responses API fallback emits the terminal event after its buffered part events; `claude_code_oauth` passes it through unchanged.
  - Contract: exactly one terminal event, always last, never after a transport error. Truncated streams emit none; usage fields absent from the wire stay `None`. Request bodies and non-streaming paths are unchanged.
- Added `CohereModel::with_base_url` for endpoint overrides (#54).
- Added `with_header(name, value)` and `with_appended_header(name, value)` builder methods to `AnthropicModel` for attaching custom HTTP headers to every request (#52):
  - `with_header()` replaces any existing value of that header, including the library's own (`x-api-key`, `anthropic-version`, `Content-Type`, `anthropic-beta`) - no header is protected.
  - `with_appended_header()` adds a value while keeping existing ones, for multi-valued headers such as `anthropic-beta`. It is the wrong choice for single-valued headers like `x-api-key`, where appending sends the header twice instead of overriding it.
  - Header names and values are trusted caller configuration, not end-user input: since no header is protected, forwarding user-controlled data into these builders would let that user overwrite `x-api-key`.
  - Header construction for the streaming and non-streaming paths is consolidated into a single `build_headers()`; an invalid header name or value now surfaces as `ModelError::Configuration` naming the header (never its value) instead of a late transport error.
- `ReasoningEffort` gains `Minimal`, `XHigh`, `Max` and `Custom(String)` variants, and `with_reasoning_effort` now accepts any string (#55):
  - `ReasoningEffort::parse` maps known strings case-insensitively; unknown strings reach the API verbatim as `Custom`, supporting newer efforts such as `xhigh` on gpt-5.1 and `max` on gpt-5.1-pro.
  - `build_model_extended` now honours `reasoning_effort` on the OpenAI branch, selecting `OpenAIResponsesModel` (the effort exists only on the Responses API) while still applying the shared `api_key`, `base_url`, `timeout` and `client` options.
- Token usage is now surfaced through the streaming `AgentStreamEvent` path (`run_stream`), matching the non-streaming `run()` path:
  - `AgentStreamEvent::ResponseComplete` now carries `usage: Option<RequestUsage>` (per-model-response token usage; `None` when the provider reports none).
  - `AgentStreamEvent::RunComplete` now carries `usage: RunUsage` (run-aggregate token usage, the field-wise sum of per-step usage).
  - `AgentStreamEvent::Cancelled` now carries `usage: RunUsage` (partial run-aggregate accumulated up to cancellation).
- Both the standard and cancellable streaming loops now accumulate per-response usage run-wide, mirroring `run()`.

### Fixed
- Streaming runs (`run_stream`) over models whose parser does not emit a terminal `StreamComplete` event no longer collapse to an empty/errored result. A change in the streaming-resilience work (#39) made the agent run-loop treat the absence of a terminal `StreamComplete` as a premature EOF, which incorrectly broke every non-Anthropic provider (OpenAI/Google/Groq/etc., whose parsers never emit `StreamComplete`) as well as in-memory test models - collapsing otherwise-valid runs to "no response". The run-loop now treats a clean stream end (content produced, no explicit error) as a successful completion (`FinishReason::Stop`). Anthropic's explicit truncation detection (`IncompleteStream` on a missing `message_stop`/partial frame) is unchanged, and the empty-stream guard still errors on a stream that produced no content.
  - Note: for non-Anthropic providers, whose parsers do not signal truncation, a genuinely truncated stream is now accepted as a normal completion (the pre-#39 behavior). Adding real termination detection for those parsers is tracked as a follow-up.
- Streaming runs (`run_stream`) against an Anthropic model no longer re-issue the same request 3-4 times for a single, tool-less completion. The agent loop ended a run only on `FinishReason::Stop`, but the Anthropic streaming parser maps normal completions to `EndTurn` (the non-streaming parser maps `end_turn` to `Stop`, so only streaming was affected). The loop now ends on the existing completeness predicate `FinishReason::is_complete()` (`Stop | EndTurn | StopSequence`), so a normal streamed completion terminates in exactly one round - eliminating the redundant model calls, latency, and token cost. Tool-call rounds are unaffected (they continue before the completion check).
  - Note: a `max_tokens`-truncated (`Length`) streamed completion is still not treated as terminal and can re-issue the request; this pre-existing behavior is now logged (via `warn!`) and tracked as a follow-up.

### Changed
- Version bump to `0.3.0` across workspace crates.
- Nix devshell exports `OPENSSL_LIB_DIR`, `OPENSSL_INCLUDE_DIR`, `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH` so `native-tls` builds work; nixpkgs and rust-overlay locks updated, moving the locked Rust stable toolchain (#53).

### Breaking
- Streams now yield one extra event: code that treats every event as content must skip `StreamComplete` (#54).
- Adding `usage` fields to the public `AgentStreamEvent` struct-variants is source-breaking for downstream code that matches them exhaustively without `..` (#51).
- Callers that previously set `reasoning_effort` on the OpenAI branch of `build_model_extended` received a chat model with the value silently dropped; they now receive a Responses model with reasoning enabled (#55).

## [0.2.6] - 2025-07-11

### Added
- Added `with_client(reqwest::Client)` builder method to all model implementations for consistent HTTP client injection:
  - `MistralModel`, `OllamaModel`, `GroqModel` (passthrough), `OpenRouterModel`, `CohereModel`, `HuggingFaceModel`, `BedrockModel`.
- Added `client: Option<reqwest::Client>` field and `with_client()` builder to `ExtendedModelConfig`.
- Wired custom client support through `build_model_extended()` for all provider branches.

### Changed
- Version bump to `0.2.6` across workspace crates.

## [0.2.5] - 2026-02-17

### Fixed
- Fixed release-breakage in `serdes-ai` version tests:
  - Replaced hard-coded version assertions with dynamic checks based on `CARGO_PKG_VERSION`.
  - Prevents future patch/minor release bumps from failing tests.

### Changed
- Version bump to `0.2.5` across workspace crates.

## [0.2.4] - 2026-02-17

### Fixed
- Follow-up clean release after `0.2.3` to ensure release branch/tag state is fully formatting- and lint-clean.
- Applied rustfmt cleanup for `serdes-ai-agent` stream test formatting on `main`.

### Changed
- Version bump to `0.2.4` across workspace crates.

## [0.2.3] - 2026-02-17

### Fixed
- Fixed tool-call argument persistence across agent history in `serdes-ai-agent`:
  - Canonicalized `ToolCallArgs` to `Json(...)` before persisting model responses in both run and stream paths.
  - Prevented malformed raw string args from being replayed to providers in subsequent requests.

### Added
- Added regression coverage in `serdes-ai-agent`:
  - Unit tests for canonicalization helper in `run.rs` and `stream.rs`.
  - End-to-end streaming test verifying `RunComplete.messages` persists canonical JSON tool-call args.

### Changed
- Version bump to `0.2.3` across workspace crates.

## [0.2.2] - 2026-02-17

### Fixed
- Fixed crates.io publish pipeline reliability:
  - Removed `|| echo "Already published, skipping"` from publish steps in `.github/workflows/publish.yml` so real publish failures fail the job.
- Fixed publish ordering/dependency deadlock by removing stale `dev-dependencies` from `serdes-ai-macros` that pulled `serdes-ai-tools` during publish validation.

### Changed
- Version bump to `0.2.2` across workspace crates.

## [0.2.1] - 2026-02-17

### Changed
- Merged dependency update PRs #18, #19, #20, #21, and #22 into `main`.
- Updated lockfile and workspace dependency set accordingly.

### Fixed
- OAuth PKCE random generation compatibility in `serdes-ai-providers`:
  - Replaced `getrandom::getrandom(...)` with `getrandom::fill(...)` for `getrandom 0.4` API compatibility.
- Ensured workspace passes CI gates after merges:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-features -- -D warnings`
  - `cargo test --workspace --all-features`

## [0.1.5] - 2025-01-28

### Added
- **Cancellation Support** for `AgentStream` and `AgentRun` (Issue #6)
  - New `AgentStream::new_with_cancel()` constructor accepting a `CancellationToken`
  - New `AgentRun::new_with_cancel()` constructor accepting a `CancellationToken`
  - New `AgentStreamEvent::Cancelled` event variant with partial results (text, thinking, pending tools)
  - `cancel()`, `is_cancelled()`, and `cancellation_token()` methods on both types
  - Cancellation checks in the streaming loop and before each tool execution
  - Re-exported `tokio_util::sync::CancellationToken` from `serdes-ai-agent` for convenience

## [0.1.2] - 2025-01-27

### Fixed
- Removed unused import `PartStartEvent` in `serdes-ai-models` claude_code_oauth stream module

## [0.1.0] - 2025-01-XX

### Added

#### Core Framework
- Complete Rust port of pydantic-ai architecture
- Type-safe agent framework with generic dependencies and output types
- Compile-time validation of agent configurations
- Async/await support throughout using tokio

#### Agent System (`serdes-ai-agent`)
- `Agent` builder with fluent API
- `AgentRun` for managing conversation state
- `RunContext` for dependency injection into tools
- Support for system prompts (static and dynamic)
- Configurable end strategies (early, first tool, exhaust tools)
- Usage tracking and limits

#### Model Providers (`serdes-ai-models`)
- **OpenAI**: Full support for GPT-4, GPT-4o, o1, o3 models
  - Chat completions API
  - Streaming with SSE
  - Tool calling with strict mode
  - Vision (image input)
- **Anthropic**: Full support for Claude 3.5/4 family
  - Messages API
  - Extended thinking (claude-3-5-sonnet)
  - Prompt caching
  - Tool use with cache control
- **Google**: Gemini 1.5 and 2.0 models
  - GenerateContent API
  - Multi-modal input (text, images, documents)
  - Native JSON mode
- **Groq**: Ultra-fast inference
  - OpenAI-compatible API wrapper
  - Llama 3, Mixtral, Gemma models
- **Mistral**: Mistral AI models
  - Native API implementation
  - Mistral Large, Small, Codestral
- **Ollama**: Local model support
  - HTTP API implementation
  - Any Ollama-compatible model
- **Azure OpenAI**: Azure-hosted OpenAI
  - OpenAI-compatible wrapper
  - Azure-specific authentication
- **AWS Bedrock**: AWS-hosted models
  - Converse API implementation
  - Claude, Llama, Titan, Mistral on AWS

#### Tool System (`serdes-ai-tools`)
- `Tool` trait for custom tool implementations
- `ToolDefinition` with JSON schema parameters
- `SchemaBuilder` for fluent schema construction
- `ToolRegistry` for tool management
- `RunContext` for dependency access in tools
- Built-in tools:
  - `FileSearchTool` - Search files by content
  - `WebSearchTool` - Web search integration
  - `CodeExecutionTool` - Safe code execution

#### Toolsets (`serdes-ai-toolsets`)
- `FunctionToolset` - Collect multiple tools
- `CombinedToolset` - Merge multiple toolsets
- `FilteredToolset` - Allow/deny tool access
- `PrefixedToolset` - Add prefixes to tool names
- `RenamedToolset` - Rename tools
- `ApprovalRequiredToolset` - Require approval for tools
- `DynamicToolset` - Add/remove tools at runtime
- `PreparedToolset` - Modify tools per-request
- `ExternalToolset` - Deferred tool execution
- `WrapperToolset` - Before/after hooks

#### Output Handling (`serdes-ai-output`)
- `OutputSchema` trait for output validation
- `StructuredOutputSchema` - JSON schema-based validation
- `TextOutputSchema` - Plain text output
- `UnionOutputSchema` - Multiple possible outputs
- JSON extraction from text responses
- Validation with detailed error messages

#### Streaming (`serdes-ai-streaming`)
- `AgentStream` for streaming responses
- `AgentStreamEvent` enum for event types
- Text delta accumulation
- Tool call streaming
- Backpressure support
- WebSocket support (for compatible providers)

#### Graph Workflows (`serdes-ai-graph`)
- `Graph` for defining workflows
- `BaseNode` trait for node implementations
- `NodeResult` for controlling flow (Next, End)
- Built-in node types:
  - `FunctionNode` - Execute async functions
  - `AgentNode` - Run agents
  - `RouterNode` - Dynamic routing
  - `ConditionalNode` - Branching
- State persistence:
  - `InMemoryPersistence`
  - `FilePersistence`
- Mermaid diagram generation
- Execution history and iteration

#### MCP Support (`serdes-ai-mcp`)
- `McpClient` for connecting to MCP servers
- `McpToolset` - Use MCP tools as toolsets
- `McpServer` for building tool servers
- JSON-RPC transport
- stdio and HTTP transports

#### Embeddings (`serdes-ai-embeddings`)
- `Embedder` trait for embedding models
- OpenAI embeddings support
- Similarity functions:
  - Cosine similarity
  - Dot product
  - Euclidean distance
  - Manhattan distance
- Vector normalization and centroid

#### Retry System (`serdes-ai-retries`)
- `RetryConfig` for retry settings
- Wait strategies:
  - Fixed delay
  - Exponential backoff
  - Jitter
- Retry conditions (rate limit, timeout, errors)
- Maximum attempts and timeouts

#### Evaluation Framework (`serdes-ai-evals`)
- `Dataset` for test cases
- `Case` for individual test scenarios
- `Evaluator` trait for custom evaluators
- Built-in evaluators:
  - Exact match
  - Contains
  - Regex match
  - LLM-as-judge
- `EvaluationReport` with statistics
- Parallel evaluation execution

#### Macros (`serdes-ai-macros`)
- `#[derive(Output)]` - Generate OutputSchema
- `#[derive(Tool)]` - Generate Tool implementation
- `#[derive(Agent)]` - Generate Agent configuration
- `#[tool]` attribute for function tools

#### Core Types (`serdes-ai-core`)
- Message types:
  - `ModelRequest` / `ModelResponse`
  - `SystemPromptPart`, `UserPromptPart`
  - `ToolCallPart`, `ToolReturnPart`
  - `ThinkingPart` for reasoning
- Content types:
  - `UserContent` (text, parts)
  - `ImageContent`, `AudioContent`, `VideoContent`
  - `DocumentContent`
- `ModelSettings` for configuration
- `RequestUsage` / `RunUsage` for token tracking
- `UsageLimits` for cost control
- Type-safe identifiers

### Technical Details

- **Minimum Rust Version**: 1.75.0
- **Async Runtime**: tokio 1.x
- **HTTP Client**: reqwest with rustls
- **Serialization**: serde + serde_json
- **Error Handling**: thiserror + anyhow
- **Tracing**: Optional tracing integration

### Dependencies

- `async-trait` - For async trait methods
- `futures` - Stream utilities
- `chrono` - Date/time handling
- `base64` - Encoding for binary content
- `uuid` - Unique identifiers
- `sha1` - Hash generation
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde` / `serde_json` - Serialization
- `thiserror` / `anyhow` - Error handling

---

## [0.1.1] - 2025-01-27

### Fixed

- **Agent Loop Premature Termination**: Fixed a bug in `serdes-ai-agent` where the agent would stop early when the model returned both text AND tool calls in the same response. With `Output = String`, any text was being treated as valid output, causing tool calls to be skipped. The fix prioritizes tool call execution over text output parsing, matching the behavior in the streaming code path. (#1)

### Changed

- Tool calls are now executed before checking for text output in `process_response`. This ensures that when a model returns explanatory text along with tool calls, the tools are always executed.

---

## [Unreleased]

### Added
- `OpenAIResponsesModel` in `serdes-ai-models` gains a WebSocket transport behind the `responses-ws` feature and opt-in conversation-keyed session chaining on both transports (#65, #66):
  - WebSocket turns send the flat `{"type":"response.create","model":...}` frame the codex CLI and [Open Responses](https://openresponses.org) servers speak and map the event stream (`output_item.added`, `output_text.delta`, `reasoning_summary_text.delta`, `function_call_arguments.delta`, ...) onto `ModelResponseStreamEvent`s, ending with exactly one terminal `StreamComplete` carrying finish reason and usage.
  - Session chaining (`with_session_chaining(true)`) keeps per-conversation state (`previous_response_id` plus the requests already delivered) and sends only each turn's new input items: the websocket holds a live socket per conversation with `store: false`, HTTP persists every turn with `store: true` and streams SSE. Stale continuations replay the full input; connection-limit and dead-socket failures reconnect. Recovery applies only before any event has reached the caller, so partial output is never duplicated.
  - Request building is wire-accurate: history maps to `instructions` plus wire `InputItems` (tool calls and returns as `function_call` / `function_call_output` items, reasoning with `encrypted_content` round-trip), and SSE streaming enforces the terminal-event contract: exactly one terminal event, always last.
  - Model errors from response envelopes map to `ModelError::Provider { code }` on both transports.
  - The wire-accurate Open Responses server that exercised the client now ships as serdes-ai-models test support (`tests/rig`, not a product surface), exercising both transports, chaining, and connection-lifetime enforcement.
  - The `codex_haiku` example moved to `serdes-ai-providers/examples` and now defaults to `wss://api.openai.com/v1/responses` with `OPENAI_API_KEY`; the codex backend (PKCE OAuth, codex headers) activates behind `--codex` or `CODEX=1`.
- `WebSocketStream::connect` in `serdes-ai-streaming` now applies configured headers to the HTTP upgrade request (auth previously silently dropped) and bounds the handshake by the configured timeout.

### Breaking
- The `serdes-ai-responses` crate is removed; the client is the `openai::responses` module of `serdes-ai-models` (`OpenAIResponsesModel`), with the WebSocket transport behind the `responses-ws` feature (#65, #66).
- The `serdes-ai` facade feature `open-responses` is replaced by `openai-responses-ws = ["serdes-ai-models/responses-ws"]`, also part of `full`; the `serdes_ai::responses` re-export is gone (#66).
- `ResponsesApiRequest.input` is now the wire `InputItems` type; the bespoke `ResponseInput` / `ResponseInputContent` / `ResponseInputPart` types are removed (#66).
- Tool returns serialize as `function_call_output` items instead of bespoke messages (#66).
- Multiple system prompts join into `instructions` instead of last-wins (#66).
- Unsupported media parts fail the request instead of being silently skipped (#66).
- Model errors from response envelopes map to `Provider { code }` on both transports (#66).

### Planned
- OpenAI Realtime API support
- Cohere provider
- Vertex AI provider
- Agent memory and conversation history
- Tool result caching
- Batch API support
- Cost estimation and tracking
- Prometheus metrics integration
