# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Planned
- OpenAI Realtime API support
- Cohere provider
- Vertex AI provider
- Agent memory and conversation history
- Tool result caching
- Batch API support
- Cost estimation and tracking
- Prometheus metrics integration
