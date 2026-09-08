//! Live smoke test for the responses model's WebSocket transport.
//!
//! Two variants share one code path:
//!
//! - **default**: `wss://api.openai.com/v1/responses`, token read from the
//!   `OPENAI_API_KEY` environment variable.
//! - **codex** (`--codex` flag or `CODEX=1`): the ChatGPT codex backend at
//!   `wss://chatgpt.com/backend-api/codex/responses`, authenticated by the
//!   OAuth PKCE flow from `serdes-ai-providers` (the codex CLI's client id,
//!   browser opens automatically, callback on localhost:1455).
//!
//! ```bash
//! # one haiku, streamed as deltas
//! cargo run -p serdes-ai-providers --example codex_haiku
//! cargo run -p serdes-ai-providers --example codex_haiku -- gpt-5.6-sol
//! # codex backend instead of the plain API endpoint
//! cargo run -p serdes-ai-providers --example codex_haiku -- --codex
//! # tool-call round trip: model calls get_weather, example answers,
//! # chained second turn returns the final answer
//! cargo run -p serdes-ai-providers --example codex_haiku -- tools
//! ```
//!
//! Codex tokens are cached in `~/.keys/.serdes_codex_token.json` (never
//! printed) and reused up to their reported expiry minus 30 seconds, capped
//! at 25 minutes. Tokens without a reported lifetime are not reused.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use futures::StreamExt;
use serdes_ai_core::messages::{
    ModelRequest, ModelRequestPart, ModelResponsePartDelta, ModelResponseStreamEvent,
    UserPromptPart,
};
use serdes_ai_core::{ClassifyModelFailure, ModelFailureKind, ModelSettings};
use serdes_ai_models::ModelError;
use serdes_ai_models::model::{Model, ModelRequestParameters};
use serdes_ai_models::openai::OpenAIResponsesModel;
use serdes_ai_models::openai::responses::Transport;
use serdes_ai_providers::{TokenResponse, chatgpt_oauth_config, run_pkce_flow};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Plain OpenAI Responses websocket endpoint (default variant).
const DEFAULT_ENDPOINT: &str = "wss://api.openai.com/v1/responses";
/// ChatGPT codex backend (`--codex` / `CODEX=1`).
const CODEX_ENDPOINT: &str = "wss://chatgpt.com/backend-api/codex/responses";
/// Default model on the plain OpenAI endpoint.
const DEFAULT_MODEL: &str = "gpt-5.1";
/// Default model on the codex backend.
const CODEX_DEFAULT_MODEL: &str = "gpt-5.6-luna";
/// Conservative reuse window; grants are typically valid for an hour.
const REUSE_SECS: u64 = 25 * 60;

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedToken {
    token: TokenResponse,
    fetched_at: u64,
}

fn token_is_fresh(cached: &CachedToken, now: u64) -> bool {
    let ttl = cached
        .token
        .expires_in
        .unwrap_or(0)
        .saturating_sub(30)
        .min(REUSE_SECS);
    now.checked_sub(cached.fetched_at)
        .is_some_and(|age| age < ttl)
}

fn write_token_cache(
    path: &std::path::Path,
    cached: &CachedToken,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec(cached)?;
    let mut random = [0_u8; 16];
    getrandom::fill(&mut random)?;
    let temporary = path.with_extension(format!("{}.tmp", URL_SAFE_NO_PAD.encode(random)));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    // Write a private new inode. Replacing a permissive old cache must not
    // expose new tokens through file descriptors already open on that cache.
    let mut file = options.open(&temporary)?;
    let result = file.write_all(&bytes);
    drop(file);
    let result = result.and_then(|()| std::fs::rename(&temporary, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result?;
    Ok(())
}

fn browser_opener(os: &str) -> &'static str {
    match os {
        "macos" => "open",
        "windows" => "explorer",
        _ => "xdg-open",
    }
}

fn parse_args(
    args: impl IntoIterator<Item = String>,
    mut codex: bool,
) -> Result<(Option<String>, bool, bool), String> {
    let mut model_name = None;
    let mut tools_mode = false;
    for arg in args {
        match arg.as_str() {
            "tools" => tools_mode = true,
            "--codex" => codex = true,
            other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
            _ if model_name.is_some() => return Err("only one model name is accepted".into()),
            _ => model_name = Some(arg),
        }
    }
    Ok((model_name, tools_mode, codex))
}

fn cache_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join(".keys")
        .join(".serdes_codex_token.json")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs()
}

async fn obtain_token() -> Result<TokenResponse, Box<dyn std::error::Error>> {
    if let Ok(raw) = std::fs::read_to_string(cache_path()) {
        if let Ok(cached) = serde_json::from_str::<CachedToken>(&raw) {
            let age = now_secs().saturating_sub(cached.fetched_at);
            if token_is_fresh(&cached, now_secs()) {
                println!("(reusing cached token, {age}s old)");
                return Ok(cached.token);
            }
        }
    }

    let config = chatgpt_oauth_config();
    let (url, handle) = run_pkce_flow(&config).await?;
    println!("Login required. Opening your browser; if nothing happens, visit:");
    println!();
    println!("  {url}");
    println!();
    println!("Waiting for the callback on localhost:1455 ...");
    std::io::stdout().flush()?;

    if let Err(error) = std::process::Command::new(browser_opener(std::env::consts::OS))
        .arg(&url)
        .spawn()
    {
        eprintln!("Could not launch a browser: {error}. Open the printed URL manually.");
    }
    let token = handle.wait_for_tokens().await?;

    let cached = CachedToken {
        token: token.clone(),
        fetched_at: now_secs(),
    };
    if let Some(dir) = cache_path().parent() {
        std::fs::create_dir_all(dir)?;
    }
    write_token_cache(&cache_path(), &cached)?;
    println!("Token cached to ~/.keys/.serdes_codex_token.json (not printed).");
    Ok(token)
}

/// Extract `chatgpt_account_id` from the id_token JWT, the header the codex
/// backend requires for ChatGPT-plan accounts. Best effort: returns None on
/// any decode failure and the request simply goes out without the header.
fn account_id(id_token: Option<&str>) -> Option<String> {
    let payload_b64 = id_token?.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    claims["https://api.openai.com/auth"]["chatgpt_account_id"]
        .as_str()
        .map(str::to_owned)
}

fn user_turn(text: &str) -> ModelRequest {
    ModelRequest::with_parts(vec![ModelRequestPart::UserPrompt(UserPromptPart::new(
        text,
    ))])
}

/// Full tool-call round trip against the live backend: register a function
/// tool, let the model call it, answer locally, and finish on a chained
/// turn. Fails loudly if any leg of the loop is broken on the wire.
async fn tool_round_trip(model: &dyn Model) -> Result<(), Box<dyn std::error::Error>> {
    use serdes_ai_core::messages::{ModelResponsePart, ToolCallArgs, ToolReturnPart};
    use serdes_ai_tools::ToolDefinition;

    let params = ModelRequestParameters::new().with_tools(vec![ToolDefinition {
        name: "get_weather".into(),
        description: "Get the current weather for a city.".into(),
        parameters_json_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
            "additionalProperties": false
        }),
        strict: Some(true),
        outer_typed_dict_key: None,
    }]);

    println!("-- turn 1: one function tool offered, asking about Tokyo weather");
    let history = vec![user_turn(
        "What is the weather in Tokyo? Call the get_weather tool.",
    )];
    let first = model
        .request(&history, &ModelSettings::default(), &params)
        .await?;

    let call = first.parts.iter().find_map(|part| match part {
        ModelResponsePart::ToolCall(call) => Some(call.clone()),
        _ => None,
    });
    let Some(call) = call else {
        let text = first
            .parts
            .iter()
            .filter_map(|part| match part {
                ModelResponsePart::Text(t) => Some(t.content.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        return Err(format!("model returned no tool call; text was: {text}").into());
    };
    let args = match &call.args {
        ToolCallArgs::String(s) => s.clone(),
        other => format!("{other:?}"),
    };
    println!(
        "-- tool call received: name={} call_id={:?}",
        call.tool_name, call.tool_call_id
    );
    println!("   arguments={args}");
    if args.trim().is_empty() {
        return Err(
            "tool call arrived with EMPTY arguments: argument deltas did not assemble".into(),
        );
    }

    // "Execute" the tool locally and return the result on a chained turn.
    let mut tool_return = ToolReturnPart::new(
        "get_weather",
        r#"{"temperature_c": 18, "conditions": "clear"}"#,
    );
    tool_return.tool_call_id = call.tool_call_id.clone();
    let history = vec![
        history.into_iter().next().expect("one turn"),
        ModelRequest::with_parts(vec![ModelRequestPart::ModelResponse(Box::new(first))]),
        ModelRequest::with_parts(vec![ModelRequestPart::ToolReturn(tool_return)]),
    ];
    println!("-- turn 2: returning the tool output on a chained turn");
    let second = model
        .request(&history, &ModelSettings::default(), &params)
        .await?;

    let text = second
        .parts
        .iter()
        .filter_map(|part| match part {
            ModelResponsePart::Text(t) => Some(t.content.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    println!();
    println!("{text}");
    println!();
    if text.trim().is_empty() {
        return Err("second turn returned no text".into());
    }
    println!(
        "-- tool round trip complete: finish={:?} usage={:?}",
        second.finish_reason, second.usage
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let (model_name, tools_mode, codex) = parse_args(
        std::env::args().skip(1),
        std::env::var("CODEX").is_ok_and(|v| v == "1"),
    )?;
    let name = model_name.unwrap_or_else(|| {
        if codex {
            CODEX_DEFAULT_MODEL.to_owned()
        } else {
            DEFAULT_MODEL.to_owned()
        }
    });

    let (model, endpoint, codex_token) = if codex {
        let token = obtain_token().await?;
        let mut model = OpenAIResponsesModel::new(&name, token.access_token.clone())
            .with_base_url(CODEX_ENDPOINT)
            .with_transport(Transport::WebSocket)
            .with_session_chaining(true)
            .with_header("Authorization", format!("Bearer {}", token.access_token))
            .with_header("OpenAI-Beta", "responses_websockets=2026-02-06")
            .with_header("originator", "serdesai_miniclient")
            .with_header("User-Agent", "serdesai_miniclient/0.1 (codex ws smoke)");
        if let Some(id) = account_id(token.id_token.as_deref()) {
            model = model.with_header("chatgpt-account-id", id);
        }
        (model, CODEX_ENDPOINT, Some(token.access_token))
    } else {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(
            |_| "OPENAI_API_KEY not set; export it or pass --codex for the ChatGPT OAuth flow",
        )?;
        let model = OpenAIResponsesModel::new(&name, api_key.clone())
            .with_base_url(DEFAULT_ENDPOINT)
            .with_transport(Transport::WebSocket)
            .with_session_chaining(true)
            .with_header("Authorization", format!("Bearer {api_key}"));
        (model, DEFAULT_ENDPOINT, None)
    };

    println!();
    println!("Connecting: model={name} endpoint={endpoint} transport=websocket");
    println!();

    let rejected_cache = codex_token.map(|token| (cache_path(), token));
    run(
        &model,
        tools_mode,
        rejected_cache
            .as_ref()
            .map(|(path, token)| (path.as_path(), token.as_str())),
    )
    .await
}

fn is_auth_rejection(error: &ModelError) -> bool {
    let failure = error.model_failure();
    if failure.kind == ModelFailureKind::Authentication || failure.status == Some(401) {
        return true;
    }
    // Responses currently classifies unrecognized provider codes as Server.
    if matches!(error, ModelError::Provider { provider, .. } if provider == "openai")
        || matches!(error, ModelError::Api { .. })
    {
        return matches!(
            failure.provider_code.as_deref(),
            Some(
                "authentication_error"
                    | "invalid_api_key"
                    | "invalid_token"
                    | "token_expired"
                    | "token_invalidated"
            )
        );
    }
    // The connector wraps Tungstenite's HTTP handshake status in a
    // StreamError::Connection string. Match that representation, not prose
    // containing "401" or generic connection failures.
    matches!(error, ModelError::Connection(message)
        if message == "Connection error: HTTP error: 401 Unauthorized")
}

fn invalidate_token_cache(
    path: &std::path::Path,
    rejected_token: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let cached: CachedToken = serde_json::from_reader(&file)?;
    if cached.token.access_token == rejected_token {
        // Truncate the opened inode, not the path: a concurrent cache writer
        // atomically renames a new inode and must not lose its fresh token.
        // An empty cache is a cache miss for obtain_token.
        file.set_len(0)?;
    }
    Ok(())
}

async fn run(
    model: &dyn Model,
    tools_mode: bool,
    codex_cache: Option<(&std::path::Path, &str)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = run_once(model, tools_mode).await;
    if let (Err(error), Some((path, token))) = (&result, codex_cache) {
        if error
            .downcast_ref::<ModelError>()
            .is_some_and(is_auth_rejection)
        {
            if let Err(cache_error) = invalidate_token_cache(path, token) {
                eprintln!(
                    "Could not invalidate the rejected Codex token cache: {cache_error}. Remove {} before running again.",
                    path.display()
                );
            } else {
                eprintln!(
                    "Codex rejected authentication. The rejected token is no longer eligible for reuse; run again to authenticate or use a newer cached token."
                );
            }
            eprintln!(
                "This invocation was not retried; output or tool effects may already have occurred."
            );
        }
    }
    result
}

async fn run_once(model: &dyn Model, tools_mode: bool) -> Result<(), Box<dyn std::error::Error>> {
    if tools_mode {
        return tool_round_trip(model).await;
    }

    let history = vec![user_turn(
        "Write us one haiku about finally getting websockets to work.",
    )];
    let mut stream = model
        .request_stream(
            &history,
            &ModelSettings::default(),
            &ModelRequestParameters::new(),
        )
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            ModelResponseStreamEvent::PartDelta(delta) => {
                if let ModelResponsePartDelta::Text(text) = delta.delta {
                    print!("{}", text.content_delta);
                    std::io::stdout().flush()?;
                }
            }
            ModelResponseStreamEvent::StreamComplete(complete) => {
                println!();
                println!();
                println!(
                    "-- stream complete: finish={:?} input_tokens={:?} output_tokens={:?}",
                    complete.finish_reason, complete.input_tokens, complete.output_tokens
                );
            }
            ModelResponseStreamEvent::PartStart(_) | ModelResponseStreamEvent::PartEnd(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serdes_ai_core::messages::{ModelResponse, ModelResponsePart};
    use serdes_ai_models::model::StreamedResponse;
    use serdes_ai_models::profile::ModelProfile;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct TestCache(PathBuf);

    impl TestCache {
        fn new() -> Self {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/pr66-final-fixes");
            std::fs::create_dir_all(&dir).unwrap();
            let mut random = [0_u8; 16];
            getrandom::fill(&mut random).unwrap();
            let path = dir.join(format!("token-{}.json", URL_SAFE_NO_PAD.encode(random)));
            let mut token = cached(Some(3600));
            token.fetched_at = now_secs();
            assert!(token_is_fresh(&token, now_secs()));
            write_token_cache(&path, &token).unwrap();
            Self(path)
        }

        fn assert_invalidated(&self, invalidated: bool) {
            let raw = std::fs::read(&self.0).unwrap();
            if invalidated {
                assert!(raw.is_empty());
            } else {
                let token: CachedToken = serde_json::from_slice(&raw).unwrap();
                assert!(token_is_fresh(&token, now_secs()));
            }
        }
    }

    impl Drop for TestCache {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }

    #[derive(Clone, Copy)]
    enum FailureAt {
        FirstToolRequest,
        SecondToolRequest,
        StreamAcquisition,
        StreamEvent,
        AfterText,
    }

    struct FailingModel {
        at: FailureAt,
        error: Mutex<Option<ModelError>>,
        calls: AtomicUsize,
        emitted: Arc<AtomicUsize>,
        profile: ModelProfile,
    }

    impl FailingModel {
        fn new(at: FailureAt, error: ModelError) -> Self {
            Self {
                at,
                error: Mutex::new(Some(error)),
                calls: AtomicUsize::new(0),
                emitted: Arc::new(AtomicUsize::new(0)),
                profile: ModelProfile::default(),
            }
        }

        fn error(&self) -> ModelError {
            self.error.lock().unwrap().take().expect("must not retry")
        }
    }

    #[async_trait::async_trait]
    impl Model for FailingModel {
        fn name(&self) -> &str {
            "fake"
        }
        fn system(&self) -> &str {
            "openai"
        }
        fn profile(&self) -> &ModelProfile {
            &self.profile
        }

        async fn request(
            &self,
            messages: &[ModelRequest],
            _: &ModelSettings,
            _: &ModelRequestParameters,
        ) -> Result<ModelResponse, ModelError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if matches!(self.at, FailureAt::SecondToolRequest) {
                if call == 0 {
                    let call = serdes_ai_core::messages::ToolCallPart::new(
                        "get_weather",
                        r#"{"city":"Tokyo"}"#,
                    )
                    .with_tool_call_id("call_weather");
                    return Ok(ModelResponse::with_parts(vec![
                        ModelResponsePart::ToolCall(call),
                    ]));
                }
                assert_eq!(call, 1, "completed tool effects must not be replayed");
                assert!(
                    messages
                        .iter()
                        .flat_map(|message| &message.parts)
                        .any(|part| {
                            matches!(part, ModelRequestPart::ToolReturn(result)
                        if result.tool_call_id.as_deref() == Some("call_weather"))
                        })
                );
            }
            Err(self.error())
        }

        async fn request_stream(
            &self,
            _: &[ModelRequest],
            _: &ModelSettings,
            _: &ModelRequestParameters,
        ) -> Result<StreamedResponse, ModelError> {
            assert_eq!(
                self.calls.fetch_add(1, Ordering::SeqCst),
                0,
                "must not retry"
            );
            if matches!(self.at, FailureAt::StreamAcquisition) {
                return Err(self.error());
            }
            let mut events = Vec::new();
            if matches!(self.at, FailureAt::AfterText) {
                events.push(Ok(ModelResponseStreamEvent::text_delta(
                    0,
                    "partial output",
                )));
            }
            events.push(Err(self.error()));
            let emitted = self.emitted.clone();
            Ok(Box::pin(futures::stream::iter(events).inspect(
                move |event| {
                    if event.is_ok() {
                        emitted.fetch_add(1, Ordering::SeqCst);
                    }
                },
            )))
        }
    }

    fn wire_auth_error() -> ModelError {
        // WsErrorEnvelope/response.failed currently retain the code but
        // classify unknown codes as Server, without HTTP status metadata.
        ModelError::provider(
            "openai",
            "token_invalidated",
            "revoked",
            ModelFailureKind::Server,
            None,
        )
    }

    #[tokio::test]
    async fn auth_rejections_invalidate_tools_and_streams_without_replay() {
        for at in [
            FailureAt::FirstToolRequest,
            FailureAt::SecondToolRequest,
            FailureAt::StreamAcquisition,
            FailureAt::StreamEvent,
            FailureAt::AfterText,
        ] {
            let cache = TestCache::new();
            let model = FailingModel::new(at, wire_auth_error());
            let expected = wire_auth_error().to_string();
            let tools = matches!(
                at,
                FailureAt::FirstToolRequest | FailureAt::SecondToolRequest
            );
            let error = run(&model, tools, Some((&cache.0, "test-access")))
                .await
                .unwrap_err();
            assert_eq!(
                error.downcast_ref::<ModelError>().unwrap().to_string(),
                expected
            );
            cache.assert_invalidated(true);
            assert_eq!(
                model.calls.load(Ordering::SeqCst),
                if matches!(at, FailureAt::SecondToolRequest) {
                    2
                } else {
                    1
                }
            );
            assert_eq!(
                model.emitted.load(Ordering::SeqCst),
                usize::from(matches!(at, FailureAt::AfterText))
            );
        }
    }

    #[tokio::test]
    async fn non_auth_errors_and_non_codex_runs_retain_cache_without_retries() {
        for tools in [false, true] {
            for codex in [false, true] {
                let cache = TestCache::new();
                let error = if codex {
                    ModelError::http(503, "unavailable")
                } else {
                    wire_auth_error()
                };
                let expected = error.to_string();
                let at = if tools {
                    FailureAt::SecondToolRequest
                } else {
                    FailureAt::AfterText
                };
                let model = FailingModel::new(at, error);
                let error = run(
                    &model,
                    tools,
                    codex.then_some((cache.0.as_path(), "test-access")),
                )
                .await
                .unwrap_err();
                assert_eq!(error.to_string(), expected);
                cache.assert_invalidated(false);
                assert_eq!(
                    model.calls.load(Ordering::SeqCst),
                    if tools { 2 } else { 1 }
                );
            }
        }
    }

    #[test]
    fn auth_detection_uses_structured_errors_and_exact_handshake_status() {
        for error in [
            ModelError::auth("rejected"),
            ModelError::http(401, "rejected"),
            ModelError::api_with_code("rejected", "authentication_error"),
            wire_auth_error(),
            ModelError::provider(
                "openai",
                "invalid_api_key",
                "rejected",
                ModelFailureKind::Server,
                None,
            ),
        ] {
            assert!(is_auth_rejection(&error), "{error}");
        }
        for error in [
            ModelError::http(403, "forbidden"),
            ModelError::http(429, "limited"),
            ModelError::http(500, "server"),
            ModelError::api("401 Unauthorized"),
            ModelError::Connection("connection reset".into()),
            ModelError::Connection("invalid header value: 401 Unauthorized".into()),
            ModelError::Connection("Connection error: HTTP error: 4010 Unauthorized".into()),
            ModelError::provider(
                "openai",
                "server_error",
                "401 Unauthorized",
                ModelFailureKind::Server,
                None,
            ),
        ] {
            assert!(!is_auth_rejection(&error), "{error}");
        }
    }

    #[test]
    fn rejected_token_does_not_invalidate_a_replacement_cache() {
        let cache = TestCache::new();
        let mut replacement = cached(Some(3600));
        replacement.fetched_at = now_secs();
        replacement.token.access_token = "new-access".into();
        write_token_cache(&cache.0, &replacement).unwrap();
        invalidate_token_cache(&cache.0, "test-access").unwrap();
        cache.assert_invalidated(false);
        let saved: CachedToken = serde_json::from_slice(&std::fs::read(&cache.0).unwrap()).unwrap();
        assert_eq!(saved.token.access_token, "new-access");
    }

    #[tokio::test]
    async fn actual_handshake_rejections_invalidate_only_401_without_retries() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for tools in [false, true] {
            for status in [
                "401 Unauthorized",
                "403 Forbidden",
                "429 Too Many Requests",
                "500 Internal Server Error",
            ] {
                let cache = TestCache::new();
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let calls = Arc::new(AtomicUsize::new(0));
                let count = calls.clone();
                let server = tokio::spawn(async move {
                    loop {
                        let (mut socket, _) = listener.accept().await.unwrap();
                        count.fetch_add(1, Ordering::SeqCst);
                        let mut request = Vec::new();
                        let mut byte = [0];
                        while !request.ends_with(b"\r\n\r\n") {
                            socket.read_exact(&mut byte).await.unwrap();
                            request.push(byte[0]);
                        }
                        socket
                            .write_all(
                                format!(
                                    "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                )
                                .as_bytes(),
                            )
                            .await
                            .unwrap();
                    }
                });
                let model = OpenAIResponsesModel::new("fake", "test-access")
                    .with_base_url(format!("ws://{addr}/v1/responses"))
                    .with_transport(Transport::WebSocket);
                let error = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    run(&model, tools, Some((&cache.0, "test-access"))),
                )
                .await
                .unwrap()
                .unwrap_err();
                let expected = format!("Connection error: HTTP error: {status}");
                assert!(matches!(error.downcast_ref::<ModelError>(),
                    Some(ModelError::Connection(message)) if message == &expected));
                cache.assert_invalidated(status.starts_with("401"));
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                server.abort();
                let _ = server.await;
            }
        }
    }

    fn cached(expires_in: Option<u64>) -> CachedToken {
        CachedToken {
            token: serde_json::from_value(serde_json::json!({
                "access_token":"test-access", "token_type":"Bearer", "expires_in":expires_in,
                "refresh_token":"test-refresh", "id_token":"test-id"
            }))
            .unwrap(),
            fetched_at: 100,
        }
    }

    #[test]
    fn cache_reuse_respects_expiry_and_clock() {
        assert!(token_is_fresh(&cached(Some(60)), 129));
        assert!(!token_is_fresh(&cached(Some(60)), 130));
        assert!(!token_is_fresh(&cached(Some(20)), 100));
        assert!(!token_is_fresh(&cached(None), 100));
        assert!(!token_is_fresh(&cached(Some(3600)), 99));
        assert!(!token_is_fresh(&cached(Some(3600)), 1600));
    }

    #[test]
    fn arguments_reject_unknown_flags_and_extra_models() {
        let parse = |args: &[&str]| parse_args(args.iter().map(|s| s.to_string()), false);
        assert!(parse(&["--codx"]).unwrap_err().contains("unknown flag"));
        assert!(parse(&["model-a", "model-b"]).is_err());
        assert_eq!(
            parse(&["tools", "--codex", "model-a"]).unwrap(),
            (Some("model-a".into()), true, true)
        );
        assert_eq!(parse(&[]).unwrap(), (None, false, false));
        assert!(parse_args(Vec::new(), true).unwrap().2);
    }

    #[test]
    fn browser_launcher_matches_platform() {
        assert_eq!(browser_opener("macos"), "open");
        assert_eq!(browser_opener("linux"), "xdg-open");
        assert_eq!(browser_opener("windows"), "explorer");
    }

    #[cfg(unix)]
    #[test]
    fn cache_write_restricts_new_and_existing_files() {
        use std::os::unix::fs::PermissionsExt;
        let dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/pr66-astra-implementation");
        std::fs::create_dir_all(&dir).unwrap();
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).unwrap();
        let path = dir.join(format!(
            "token-cache-test-{}.json",
            URL_SAFE_NO_PAD.encode(random)
        ));
        write_token_cache(&path, &cached(Some(60))).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        let mut old_file = std::fs::File::open(&path).unwrap();
        write_token_cache(&path, &cached(Some(120))).unwrap();
        let mut old_bytes = Vec::new();
        std::io::Read::read_to_end(&mut old_file, &mut old_bytes).unwrap();
        let old: CachedToken = serde_json::from_slice(&old_bytes).unwrap();
        assert_eq!(
            old.token.expires_in,
            Some(60),
            "old descriptors cannot read replacement credentials"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let saved: CachedToken = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved.token.expires_in, Some(120));
        std::fs::remove_file(path).unwrap();
    }
}
