# Reliability branch: authoritative current status

Owner: planning-agent-a12d28. Worktree: /Users/gabe/repos/fedstew/serdesAI.
Branch: feat/reliable-streaming-and-checkpoints. HEAD/base:
d0aae194fc6a8bf98bd9c1f6206099af4c126046. All work remains uncommitted;
prior dirty/untracked changes retained. No delegation, commits, publication,
real-provider calls, or application changes.

## Typed validated streaming output

Agent::run_stream_typed(prompt, deps, RunOptions) returns TypedAgentStream<Output>.
It yields the same AgentStreamEvent values as the backward-compatible AgentStream.
Use finish().await to drain with backpressure and recover Result<Option<Output>,
AgentRunError>, or take_output() after completion to retrieve the value once.
The value is the actual schema/validator-transformed Output, not reparsed raw text.
No Clone/Serialize bound is added to Output. Values are published after terminal
checkpoint success; Length/ContentFilter partials return None, not validated output.
The facade reexports TypedAgentStream. Existing stream APIs remain source-compatible.

## Tool execution and output decisions

Streaming batches honor parallel_tool_calls and max_concurrent_tools; buffered
ordered futures bound concurrency and retain result order. Zero concurrency is
clamped to one in streaming and nonstreaming, avoiding a deadlock. Tool retries
obey per-tool limits and use the run's context/call IDs. Dropping a batch drops
in-flight futures, but external effects remain potentially committed/unknown.
AfterResponse is awaited before dispatch and AfterTools after complete batches.

An explicit native output-tool plus ordinary tools is now handled in BOTH run.rs
and streaming: validate final output first; Early skips ordinary side effects;
Exhaustive executes them then finishes with the validated output. Skipped/output
calls receive protocol acknowledgements. Validation rejection does not execute
ordinary tools and retries with protocol-associated results. Plain explanatory
text plus ordinary tools retains existing tool-first behavior. This is an
intentional semantic correction for explicit output tools, not blanket text-first
execution. Error/retry text and some history acknowledgement strings still differ
between paths; complete byte-for-byte history parity is not claimed.

## Native Responses indexed output and metadata

Message content slots map to distinct native text parts. Slots must be declared
contiguously (0..N); deltas can interleave afterward without scrambling array order.
A real localhost HTTP fixture emits slot 1 before slot 0's text and verifies the
accumulated first/second parts. Slot count is bounded to 1025 per item. Unsupported
message content types fail explicitly; opaque top-level built-in/image output
records are archival only and never authorize application tools/replay.

Reasoning slot deltas are retained per summary_index and checked against the
ordered item snapshot. Slot zero streams immediately; later slots are buffered
until the snapshot to avoid corrupting the single encrypted reasoning block.
The exact summary array/encrypted content remains in native metadata. Annotation
and refusal slot metadata are emitted from item snapshots; individual annotation/
refusal delta events are not yet exposed incrementally. Repeated completed item
snapshots do not append duplicate text/signature metadata.

Optional StreamCompleteEvent.metadata carries response_id, actual model, status,
created_at/service_tier/incomplete_details, native usage and ordered output_records.
It flows to ModelResponse vendor fields and Agent checkpoints. Both accumulator
helpers offer handle_stream_complete; callers using get_parts alone must adopt the
terminal handler/get_response API to preserve response-level metadata. Retained
item/terminal metadata rejects >2 MiB. RequestUsage retains detailed usage; agent
RunUsage.token_totals_complete distinguishes complete counts from lower bounds.
Empty output retains IDs even if output validation subsequently rejects it.
TerminalMetadata, ModelResponse and text/thinking delta Debug are redacted.

## Lifecycle semantics

Both stream constructors use one worker supervisor observing explicit cancellation
and receiver closure through model setup, idle reads, policies, summary calls,
validators, tools and channel sends. Successful terminal checkpoint wins over
later detach/cancel. Sink rejection/timeout is never recursively saved. A new
synchronized test cancels during an in-progress BeforeRequest save, releases it,
and verifies no model call and exactly one Cancelled checkpoint afterward.
Earlier detach/request/summary/tool/full-channel/deadline tests remain green.

No async Drop persistence task is launched. Runtime shutdown, panic/forced abort
and blocking callbacks cannot guarantee awaited persistence. External effects
cannot be rolled back. Nonstreaming steps now use a task-local save scope: cancellation outside a save
cancels work promptly; cancellation during a save awaits its configured timeout.
A completed nonterminal save checks cancellation before model/tool work proceeds.
Successful terminal save wins; rejection/timeout returns Checkpoint error without
recursive persistence. Completed/failed runs cannot write another terminal outcome
on repeated step(). Direct future drop/task abort remains outside cooperative
cancellation guarantees. Deterministic tests cover both BeforeRequest and Terminal
saves with cancellation plus success, rejection, and timeout (six cases).

## Validation on latest code

- cargo test --workspace: 1525 passed, 0 failed, 106 ignored.
- cargo check --workspace: passed.
- cargo check -p serdes-ai-models --no-default-features --features
  openai,anthropic,azure,groq,openrouter: passed.
- cargo fmt --all -- --check and git diff --check: passed.

New paired real-Agent fixtures: transformed typed output vs run(), mixed explicit
output-tool Early/Exhaustive side effects, bounded concurrency at 1 and 2.
Native HTTP Agent fixtures: empty-output ID, actual provider model, refusal,
encrypted reasoning, detailed metadata and checkpoint serde roundtrip. Native
chunked HTTP fixture: interleaved message slots retain semantic part order.
Existing native provider interruption, usage and replay regressions remain green.
No all-features/AWS, live API, app opt-in or adapter-retirement claim.

## Remaining concrete limitations

- No incremental later-summary-slot rendering or annotation/refusal delta API;
  exact data arrives at item/terminal snapshots. Cross-item content declarations
  that arrive out of semantic order are not fully supported.
- Supported Responses message/text/refusal and reasoning records now replay from
  one shared whitelist conversion over ordered native metadata, preserving content
  arrays and one message ID per original item. A localhost test proves exact
  stream/nonstream/serde request equality. Function replay requires a corresponding
  canonical typed call; opaque built-in/image/audio records are never replayed.
  Callers editing response parts must also discard stale vendor_details before
  replay; native metadata is authoritative for passive content grouping.
- No exhaustive provider/retry/history-order parity suite beyond
  the new explicit fixtures; not every public EndStrategy/error case is proven.
- Cross-item descending declarations fail with typed InvalidResponse, preserving
  already emitted partials and emitting no completion or new callable part.
- Malformed mixed output-tool retry tests prove both paths retain preceding calls,
  matching result IDs/order, and execute no ordinary side effect before correction.
- No automatic resume, transaction journal or exactly-once side-effect guarantee.
- Images/audio are opaque archival records, not decoded/rendered generated output.

Source-breaking additions from the branch: StreamCompleteEvent.metadata and
RunUsage.token_totals_complete struct fields; lifecycle/compression enum variants.
Serde defaults accept old serialized data. See CHANGELOG for preceding additions.
New typed API is additive. No dependency/app patch was applied; test an app only
in a disposable worktree with compatible local Cargo patch versions.

---
# Historical implementation notes (not the current contract)

# Current status: unified streaming lifetime and real output validation

Owner: planning-agent-a12d28. Implementation: code-puppy-6b924b.
Worktree /Users/gabe/repos/fedstew/serdesAI; branch
feat/reliable-streaming-and-checkpoints; base/HEAD
 d0aae194fc6a8bf98bd9c1f6206099af4c126046.
Earlier dirty/untracked work was retained. No commits, fetch, delegation, pushes,
publication, live providers, or application/dependency edits in this pass.

## Implemented in this pass

Both streaming constructors now use ONE worker loop. stream_lifecycle.rs observes
receiver closure and cancellation across the entire worker future, not just sends.
This includes RunStart, final delivery, request setup, idle streams, tools, custom
policies, dynamic prompts, validators and legacy summaries. Before work starts the
supervisor retains initial history/prompt. During streaming it retains native
parts/metadata before downstream delivery. ConsumerDetached and ValidationFailed
are new CheckpointBoundary variants (exhaustive-match source breaking).

A successful Terminal checkpoint wins over later cancellation/detachment. Before
that commit, detach records ConsumerDetached; explicit cancellation records
Cancelled. If both signals are ready on the same poll, detach wins. Final error
notification is best effort and never waits for channel space. Returned worker
failure without another terminal snapshot gets a Failed checkpoint. Already
recorded terminal boundaries and sink failures suppress another save attempt.

An in-progress sink save is polled through its configured deadline before the
worker is dropped. A successful save cannot advance to another side effect while
interruption finalization is pending. Sink rejection/deadline is never recursively
saved. There is no async Drop or detached persistence task. Runtime shutdown,
process termination, panic, external task abort and blocking user callbacks cannot
guarantee an awaited checkpoint; synchronous RAII cleanup is the only guarantee
available when futures are dropped. External side effects cannot be undone.

Legacy summary calls have a deadline (model_settings.timeout, or 30 seconds).
Successful summary usage is folded once into aggregate usage; failed requests
count as requests without inventing tokens. Limits are rechecked before the main
request. Summarize now stops on failed/empty summaries. Use the new explicit
SummarizeOrTruncate compatibility strategy to allow fallback. Existing small-
history/nothing-to-compress paths retain their legacy behavior. CompressionStrategy
has a new source-breaking enum variant; ContextCompression literals are unchanged.

Streaming now invokes the configured OutputSchema and async OutputValidator chain
with shared deps, run ID, settings, metadata and retry count. Dynamic prompts and
instructions run once per run and survive validation retries. Parsing/validation
failures retain rejected native responses, add retry protocol messages, and obey
max_output_retries. Exhaustion records ValidationFailed and cannot emit OutputReady
or RunComplete. Retry prompts deliberately use generic redacted diagnostics.
Output-tool calls are parsed as output, not dispatched as application tools.
Ordinary tools retain priority over accompanying output, matching current run.rs.
Tool contexts now preserve run identity and retryable tool errors obey max_retries.
AfterResponse checkpoints include current response usage. Length/content-filter
partials still complete as partial runs but no longer emit OutputReady.

## Important remaining gaps — not full parity / not adapter retirement ready

- AgentStream remains type-erased: validators actually execute, but transformed
  typed output is not exposed by OutputReady. Use non-streaming run for the typed
  result. A generic stream result API remains future work.
- Streaming tools remain sequential; parallel_tool_calls/max_concurrent_tools
  parity is NOT implemented in this pass. Mixed output-tool plus ordinary-tool
  responses still need protocol acknowledgement/parity fixtures. EndStrategy
  follows ordinary-tool priority but exhaustive multi-output behavior is not
  comprehensively verified against run.rs. These are unresolved, not claimed done.
- Partial Length/ContentFilter terminal runs bypass output validation by design;
  RunComplete means the partial run ended, not a validated typed output. Consumers
  must use OutputReady to distinguish validated completion.
- No exact-once external execution, automatic restore, panic-safe async journal,
  or durability guarantee beyond the app's sink. The full requested cancellation
  matrix (including every sink race and structured tool retry combination) is not
  complete; see actual tests rather than assuming blanket coverage.
- Richer Responses output metadata is intentionally left to the next pass.
- Existing mock/third-party stream EOF compatibility and response retention
  limitations remain. The native OpenAI parsers enforce their own strict EOF.

## Validation (latest source)

cargo test --workspace: 1512 passed, 0 failed, 106 ignored.
cargo check --workspace: passed.
cargo check -p serdes-ai-models --no-default-features --features
openai,anthropic,azure,groq,openrouter: passed.
cargo fmt --all -- --check and git diff --check: passed.
No all-features/AWS or real-provider execution claimed.

New stream_parity integration tests exercise actual AgentStream constructors:
async validator retry + dynamic prompt/deps, structured schema exhaustion,
pre-start detach, idle partial detach, terminal winner, pending request and legacy
summary cleanup, and pending async tool cleanup without another request. A focused
supervisor test fills a one-slot final channel then cancels after terminal commit.
Earlier checkpoint interval/sink timeout/full-channel tests remain green.

Changed modules: agent.rs/builder.rs share schema, validator and prompt objects via
Arc internally without changing builder call signatures; stream.rs removes the
duplicate implementation; stream_lifecycle.rs and stream_output.rs isolate new
behavior; lifecycle.rs/run.rs add documented enum semantics. Tests are in
serdes-ai-agent/tests/stream_parity.rs. The large preexisting stream.rs was reduced
substantially rather than mechanically split; new helper files are below 600 lines.

---
# Historical notes below (superseded wherever they conflict with current status)

# Latest status: native Responses SSE and lifecycle hardening

This section supersedes earlier chronological pass notes below. All changes are
local/uncommitted on feat/reliable-streaming-and-checkpoints at the original
worktree/base. No app checkout or dependencies changed.

## Native Responses now streams

`openai/responses_stream.rs` incrementally parses the actual HTTP bytes_stream;
`request_stream` sends stream:true and never calls the buffered request method.
SSE records support CRLF, multiline data and arbitrary UTF-8 byte boundaries.
Text, reasoning summary and function-argument deltas emit immediately. Native
reasoning items retain IDs, summary arrays and encrypted_content, including empty
summaries. IDs are preserved in part provider_details (response_id/item_id).
Item order is checked; function JSON is validated before completed success.

Terminal response.completed is required for safe function dispatch; incomplete
max_output_tokens maps to Length, content_filter maps separately; unknown reasons
fail closed. response.failed/error and response.cancelled are errors, EOF before a
terminal record is IncompleteStream. No error is followed by a success terminal.
Both Agent streaming loops skip tools for Length and ContentFilter. Non-streaming
request remains JSON HTTP; incomplete_details now uses the same explicit reasons.

Scope: text/reasoning/function-call Responses events, not complete audio/image or
built-in-tool telemetry. Multiple text content slots are concatenated in their
arrival order within each native item. Unknown telemetry events are ignored;
unknown terminal incomplete reasons fail. No raw SSE is logged. Thinking delta
Debug now redacts signatures/content. There is a 16 MiB pending-record cap, not a
whole-response memory cap. Terminal-only responses are accepted; this is distinct
from buffering the HTTP transport. Response IDs are in part metadata, not a new
StreamComplete field; empty-output response IDs are not surfaced yet. Detailed
reasoning-token counters beyond standard usage fields are not exposed by the
existing terminal event type.

## Lifecycle additions

CheckpointSink now has default methods partial_interval() (None by default) and
save_timeout() (30 seconds). Apps can request 300ms Partial snapshots while a
stream is active or idle. Intervals clamp to >=10ms and missed ticks delay rather
than burst. Save is awaited with backpressure, not a background unbounded queue.
The framework does not claim the application sink is disk-durable.

Save timeout stops the run without retry or further tool/model calls. A timeout
may leave an application write committed: reconcile before resume. Streaming
cancellation does not preempt save, so latency is bounded by save_timeout for
cooperative async implementations. Blocking synchronous sink work cannot be
preempted by Tokio. Non-streaming whole-step cancellation may still preempt save.

Cancellable streaming now selects cancellation during custom policy preparation
and ordinary blocked event sends. Current native partial state is captured before
awaiting consumer delivery. Cancellation/error notification uses try_send where
needed: a saturated/dropped consumer may not receive that notification, but the
awaited checkpoint remains authoritative. Failure checkpoints added for context
policy errors, usage-limit exits and empty response EOF; sink failure never
recursively checkpoints. Before-tool cancellation now also checkpoints.

Residual lifecycle gaps: legacy automatic summarization is not cancellation-
selected; RunStart/final RunComplete delivery still uses legacy channel behavior;
receiver-drop exits are not universally journaled; no transactional tool journal,
exactly-once replay or automatic resume; streaming output-validator/dynamic-prompt
parity is still incomplete. Normal tool errors remain model-visible retry results,
not terminal run failure. Non-streaming response archive retention remains legacy.
No claim is made that every possible early exit has a terminal snapshot yet.

Facade exports are available at serdes_ai::{AgentCheckpoint, CheckpointBoundary,
CheckpointSink, ContextPolicy, ContextPolicyInput, ContextFailurePolicy}, as well
as serdes_ai::agent module. Existing durability_hook example still compiles.

## New regression coverage

- models/tests/responses_sse.rs: actual localhost HTTP/1.1 chunked SSE with
  one-byte chunks, completed/length/content-filter, EOF/malformed/failed/cancelled,
  encrypted empty-summary reasoning and fragmented function arguments.
- agent/tests/responses_native_agent.rs: actual native Responses Agent only
  dispatches tools after completed; EOF never dispatches.
- agent/tests/checkpoint_interval.rs: idle partial snapshots, full-channel
  cancellation, pending policy cancellation, sink deadline without recursive save
  or model calls.
- Existing JSON reasoning replay and provider/lifecycle safety suites retained;
  former buffered stream fixtures now serve SSE terminal records.

Commands: cargo test --workspace; cargo check --workspace; cargo fmt --all --
--check; cargo check -p serdes-ai-models --no-default-features --features
openai,anthropic,azure,groq,openrouter; cargo run -p serdes-ai-agent --example
durability_hook. No all-features/AWS or live-provider claims.

## Testing the application without changing its checkout

Not attempted this pass. Use a disposable app copy/worktree, then add temporary
[patch.crates-io] paths for every participating serdes-ai crate from this workspace.
The app's exact 0.2.6 requirements must also be adjusted in that disposable copy
to compatible 0.3.0 requirements: Cargo patches cannot override incompatible
version constraints. Inspect cargo tree for duplicate framework versions before
running native_provider_gate. Those tests characterize published defects and
need positive expectation updates for the branch. Delete the disposable copy
when done; do not edit the real app manifest/lockfile. Adapter removal still
requires application-level verification and resolution of the limits above.

---

# Native provider reliability: local review notes

Owner: planning-agent-a12d28. Implementation: code-puppy-87bdd6.

## Checkout

- Worktree: `/Users/gabe/repos/fedstew/serdesAI` (original worktree).
- Branch: `feat/reliable-streaming-and-checkpoints`.
- Base: `d0aae194fc6a8bf98bd9c1f6206099af4c126046`; local `main` and
  `origin/main` agreed. Initial checkout was clean; no repository/ancestor
  AGENTS.md was found. No fetch, stash, reset, commits, pushes or publishing.
- Workspace version is 0.3.0. These are **uncommitted, unpublished** changes.

## Implemented contracts

Chat defaults to `[DONE]` as transport completion, not EOF. A finish frame
preserves `Length`/other existing typed reasons but does not bypass trailing
usage reception. Compatible endpoints may explicitly opt into finish-frame plus
clean EOF with `with_finish_reason_terminal(true)`. Transport errors remain
errors even with that option. Errors never produce a later success terminal.
Dropping a stream releases its inner HTTP stream; it does not fabricate a
cancellation/success event after the consumer has gone away.

The parser buffers bytes through UTF-8 boundaries, queues all events from one
frame, handles no-space `data:` and CRLF, closes parts in index order, and does
not concatenate alternative choices into one response. Malformed JSON/schema
frames fail without logging raw payloads. Unknown Chat finish strings retain the
existing Stop mapping. SSE multiline-data records and resource-size limits are
not implemented in this pass.

Usage-only frames populate the terminal event; missing usage remains None.
Request/aggregate usage code already existed locally and was not redesigned.
Aggregate completeness across mixed known/unknown requests is still a follow-up.
The request's stream_usage flag is now honored, including false.

Chat cap selection uses the model profile plus o4/GPT-5 names; explicit true/false
builder override handles deployment aliases and legacy-compatible servers.
Tools, output schema, custom clients, URLs and headers retain existing plumbing.

Responses **still uses its buffered HTTP request fallback**. It now rejects
cancelled and in-progress results, preserves reasoning IDs, exact summary item
boundaries and encrypted_content (in ThinkingPart.signature), and emits native
reasoning/function-call/function-call-output replay items. It requests
reasoning.encrypted_content. Thinking and raw Responses items redact Debug
payloads. No invented Chat signature fields were introduced. Anthropic already
has native signature deltas and truncation checks; its existing suite passed.
Unknown reasoning fields are not yet retained.

Both agent stream loops treat Length as terminal partial output and skip tools
on that response. Native parser errors short-circuit before tool execution;
localhost Agent tests verify both interrupted tools and token-limit partial tools.
The agent's legacy clean-EOF fallback for other/mock models remains unchanged.

## API compatibility

Additive model builders are source-compatible. Error behavior is intentionally
stricter. The public ResponseInput enum gains Item; ResponsesApiRequest gains
include; ResponseOutputItem::Reasoning gains encrypted_content. Exhaustive
matches and literals may require edits (source-breaking). Debug formatting is
intentionally changed. A shared Chat parser means inherited OpenAI-compatible
adapters also get strict EOF behavior; wrapper-specific opt-in forwarding is not
added here. Review release/version policy before merging.

The Chat parser's existing tests moved to stream_tests.rs; both files are under
600 lines. Existing large response/core/agent files were kept cohesive rather
than split by arbitrary line count.

## Reproducible validation (no provider credentials)

Passed locally:

```sh
cargo fmt --all -- --check
cargo check --workspace --offline
cargo check -p serdes-ai-models --no-default-features --offline
cargo check -p serdes-ai-models --no-default-features --features openai,anthropic,azure,groq,openrouter --offline
cargo test -p serdes-ai-models --no-default-features --features openai,anthropic
cargo test -p serdes-ai-core -p serdes-ai-streaming -p serdes-ai-agent
cargo test -p serdes-ai-models -p serdes-ai-agent --tests
cargo test -p serdes-ai-agent --test native_stream_interruption
```

No-feature check has an existing unused-import warning in models/lib.rs.
New integration files exercise actual localhost native models, cap request keys,
usage-only frames, malformed/provider errors, EOF, native encrypted replay,
cancelled Responses and agent dispatch safety. Parser-level tests exercise every
byte split and drop cancellation; they are not a real socket cancellation test.

```sh
cargo test --workspace --offline
```

Blocked before execution by uncached `anes v0.1.6`; no network dependency download
was attempted for this command. Full all-features/AWS matrix was not run.

## Remaining before retiring the app adapter

1. Implement actual Responses SSE with response.completed/incomplete/failed
   terminal handling, preserving emitted partials and usage/metadata.
2. Complete Responses incomplete_details mapping (currently every incomplete
   response maps to Length), and preserve partial tool arguments without inventing
   valid arguments on buffered incomplete responses.
3. Add new Anthropic native HTTP signature replay coverage, unknown metadata
   preservation policy, and full aggregate-usage completeness tracking.
4. Native socket cancellation and cancellable-agent integration coverage;
   current drop test covers parser lifetime only.
5. Agent lifecycle hooks/checkpoints, resumability and remaining non-Chat provider
   EOF contracts. No app SQLite/UI or production changes belong in this PR.

This is a meaningful Chat repair and buffered Responses metadata improvement,
not completion of the entire reliability/checkpoint roadmap.

## Agent lifecycle pass (same branch, still uncommitted)

Public APIs added in `serdes-ai-agent/src/lifecycle.rs`, re-exported at crate root:

- `ContextPolicy<Deps>::prepare(ContextPolicyInput, Vec<ModelRequest>)`;
  `ContextPolicyInput` exposes RunContext, request settings/parameters (tools and
  schema) and actual model/profile/context window. Policies own independent
  summary models/tokenizers; no app tokenizer is hardwired.
- `ContextFailurePolicy::{Stop, KeepHistory}`. Stop is default; KeepHistory is
  explicit opt-in, retaining the processor output before the failed policy.
- Builder `context_policy`, `context_failure_policy`, `checkpoint_sink`.
- Existing `history_processor` is unchanged publicly and acts as the infallible
  compatibility path. Its collection now uses Arc internally so spawned runs can
  apply every processor before every primary model request.
- `CheckpointSink::save(&AgentCheckpoint)` is asynchronous and awaited.
- `AgentCheckpoint` version 1 stores run/step/boundary, canonical messages,
  optional current/partial ModelResponse and RunUsage. `from_json` checks version
  and only deserializes: it never dispatches tools or resumes a run.
- `CheckpointBoundary`: BeforeRequest, AfterResponse, AfterTools, Terminal,
  Cancelled, Failed, ModelFailed(ModelFailureKind). Provider failure snapshots
  retain typed classification without raw provider error strings.
- `AgentRunError::{ContextPolicy, Checkpoint}` are new variants, source-breaking
  for exhaustive downstream matches. RunUsage gains serde support.

### Guarantee and scope

All three request loops now replace canonical active messages with processor /
policy output. Attached processors or policy disable streaming legacy automatic
compression, avoiding double compaction. Custom policy failure does not silently
truncate. Native message blocks are retained structurally; configured policy
outputs are checked for orphan/unresolved ID-bearing tool pairs. Policies remain
responsible for preserving signatures and intentionally dropping whole groups;
validation does not prove arbitrary application transformations semantically safe.
Legacy no-policy compression behavior is unchanged, including its historical
summary fallback. Non-streaming result.responses remains its existing archive;
streaming keeps only the latest response outside compacted active history.

BeforeRequest is saved after preparation and before HTTP; AfterResponse is saved
before tool dispatch; AfterTools follows committed batch returns; Terminal is
saved before RunComplete. Sink rejection stops further model/tool calls and is
not retried automatically. Streaming failure/cancellation snapshots retain raw
partial text, thinking/signatures and tool argument fragments, not repaired JSON.
No snapshot is persisted on every token. Snapshot Debug redacts payloads; JSON is
sensitive and storage encryption/access policy belongs to the application.

Cancellable streaming now selects cancellation during request establishment and
pending async tools as well as stream reads. Cancellation drops the owned future
(RAII cleanup runs); it cannot undo an external effect or await arbitrary async
Drop cleanup. A cancelled tool is not emitted as ToolExecuted success or appended
as a completed batch. Applications must reconcile ambiguous external effects
before manually resuming. Non-streaming step cancellation now covers the whole
step and emits an interrupted boundary; it may interrupt a sink save itself.
Awaited sink durability is intentionally not preempted in the streaming loops.

### Still not guaranteed / follow-up

- Streaming context-policy preparation, legacy compression and blocked consumer
  event sends are not yet cancellation-selected. Ordinary `run_stream` has no
  cancellation token; use public `AgentStream::new_with_cancel` for cancellation.
- Legacy clean-EOF fallback for other/mock providers remains. Native Chat and
  Anthropic errors are safe; this is not universal provider terminal enforcement.
- Not every early exit (usage/output validation/context-policy failures) currently
  emits a failure checkpoint in streaming; those remain typed returned errors.
- There is no atomic transaction between remote tool side effects and sink save,
  per-tool in-flight journal, automatic replay, or automatic restore API.
  AfterTools is a batch boundary; cancellation within a batch may require external
  reconciliation of earlier tools. Do not infer exactly-once execution.
- Snapshot aggregate usage inherits existing unknown-count limitations; response
  usage is Option and stays unknown when absent.
- Dynamic prompt parity, configurable non-streaming response archive retention,
  an explicit remaining-budget field beyond model profile/settings, and complete
  native Responses SSE are not implemented in this pass.

### Tests and example

`serdes-ai-agent/tests/lifecycle_hooks.rs`: 8 passing tests covering canonical
processor/checkpoint parity, repeated tool iterations with shrinking active
history, explicit policy failure behavior, sink failure before tools, native
partial/signature roundtrip and redacted Debug, typed partial EOF snapshot,
localhost delayed request cancellation, stream cancellation and pending tool
cleanup without successful completion.

`serdes-ai-agent/examples/durability_hook.rs` demonstrates an app-owned sink with
no database coupling and deserialization without replay. Its in-memory example
store is not claimed to provide durable storage.

Passed after this pass:

```sh
cargo test -p serdes-ai-agent
cargo test -p serdes-ai-agent --test lifecycle_hooks
cargo test --workspace
cargo check --workspace
cargo fmt --all -- --check
cargo run -p serdes-ai-agent --example durability_hook
git diff --check
```

The earlier offline `anes` blocker is resolved: public dev dependencies were
downloaded and the full default-feature workspace test suite passed. This does
not mean all-features/AWS or real-provider testing. Earlier provider patches and
native safety tests remain in place. Adapter retirement is still **not ready**.
