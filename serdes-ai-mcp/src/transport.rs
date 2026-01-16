//! MCP transport implementations.
//!
//! This module provides transport abstractions for MCP communication.

use crate::error::{McpError, McpResult};
use crate::types::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{oneshot, Mutex};

/// Trait for MCP transport implementations.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a request and wait for response.
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse>;

    /// Send a notification (no response expected).
    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()>;

    /// Close the transport.
    async fn close(&self) -> McpResult<()>;

    /// Check if the transport is connected.
    fn is_connected(&self) -> bool;
}

/// Stdio transport for local MCP servers.
///
/// This transport communicates with an MCP server via stdin/stdout.
pub struct StdioTransport {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl StdioTransport {
    /// Spawn a new process and connect via stdio.
    pub async fn spawn(command: &str, args: &[&str]) -> McpResult<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| McpError::Transport(format!("Failed to spawn {}: {}", command, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("No stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("No stdout".to_string()))?;

        let stderr = child.stderr.take();

        let pending = Arc::new(Mutex::new(HashMap::new()));

        // Spawn reader task for stdout
        tokio::spawn(Self::reader_task(stdout, pending.clone()));

        // Spawn stderr drainer to prevent blocking if server writes to stderr
        if let Some(stderr) = stderr {
            tokio::spawn(Self::stderr_drainer(stderr));
        }

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(stdin)),
            pending,
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Spawn a new process with custom environment variables and connect via stdio.
    ///
    /// This method allows passing environment variables to the spawned process.
    /// The provided environment variables are merged with the parent process environment,
    /// with child env vars overriding parent env vars in case of conflicts.
    pub async fn spawn_with_env(
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
    ) -> McpResult<Self> {
        let mut child = Command::new(command)
            .args(args)
            .envs(env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| McpError::Transport(format!("Failed to spawn {}: {}", command, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("No stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("No stdout".to_string()))?;

        let stderr = child.stderr.take();

        let pending = Arc::new(Mutex::new(HashMap::new()));

        // Spawn reader task for stdout
        tokio::spawn(Self::reader_task(stdout, pending.clone()));

        // Spawn stderr drainer to prevent blocking if server writes to stderr
        if let Some(stderr) = stderr {
            tokio::spawn(Self::stderr_drainer(stderr));
        }

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(stdin)),
            pending,
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    async fn reader_task(
        stdout: ChildStdout,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    ) {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let response: JsonRpcResponse = match serde_json::from_str(trimmed) {
                        Ok(resp) => resp,
                        Err(_) => continue,
                    };

                    let request_id = match &response.id {
                        RequestId::Number(id) if *id >= 0 => Some(*id as u64),
                        _ => None,
                    };

                    if let Some(id) = request_id {
                        let sender = {
                            let mut pending = pending.lock().await;
                            pending.remove(&id)
                        };

                        if let Some(tx) = sender {
                            let _ = tx.send(response);
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Drain stderr to prevent the process from blocking.
    ///
    /// Some MCP servers (especially those launched via npx) write to stderr.
    /// If we don't read from stderr, the pipe buffer fills up and blocks the process.
    async fn stderr_drainer(stderr: ChildStderr) {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {}     // Discard stderr output
                Err(_) => break,
            }
        }
    }

    async fn send_raw(&self, data: &str) -> McpResult<()> {
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(data.as_bytes())
            .await
            .map_err(McpError::Io)?;
        stdin.write_all(b"\n").await.map_err(McpError::Io)?;
        stdin.flush().await.map_err(McpError::Io)?;
        Ok(())
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        let request_id = match &request.id {
            RequestId::Number(id) if *id >= 0 => *id as u64,
            RequestId::Number(_) => {
                return Err(McpError::Transport(
                    "Negative request IDs are not supported over stdio".to_string(),
                ));
            }
            RequestId::String(_) => {
                return Err(McpError::Transport(
                    "String request IDs are not supported over stdio".to_string(),
                ));
            }
        };

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(request_id, tx);

        let json = serde_json::to_string(request)?;
        if let Err(err) = self.send_raw(&json).await {
            self.pending.lock().await.remove(&request_id);
            return Err(err);
        }

        rx.await.map_err(|_| McpError::ConnectionClosed)
    }

    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()> {
        let json = serde_json::to_string(notification)?;
        self.send_raw(&json).await
    }

    async fn close(&self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);

        let mut child = self.child.lock().await;
        if let Some(mut c) = child.take() {
            c.kill().await.ok();
        }
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// HTTP transport for remote MCP servers.
///
/// This transport communicates with an MCP server via HTTP.
#[cfg(feature = "reqwest")]
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    session_id: Arc<Mutex<Option<String>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
    custom_headers: HashMap<String, String>,
}

#[cfg(feature = "reqwest")]
impl HttpTransport {
    /// Create a new HTTP transport.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            session_id: Arc::new(Mutex::new(None)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            custom_headers: HashMap::new(),
        }
    }

    /// Create with custom client.
    pub fn with_client(client: reqwest::Client, base_url: impl Into<String>) -> Self {
        Self {
            client,
            base_url: base_url.into(),
            session_id: Arc::new(Mutex::new(None)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            custom_headers: HashMap::new(),
        }
    }

    /// Create with custom headers (e.g., Authorization).
    pub fn with_headers(base_url: impl Into<String>, headers: HashMap<String, String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            session_id: Arc::new(Mutex::new(None)),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            custom_headers: headers,
        }
    }
}

/// Parse SSE response format
#[cfg(feature = "reqwest")]
fn parse_sse_response(text: &str) -> McpResult<String> {
    // SSE format:
    // event: message
    // data: {"jsonrpc":"2.0",...}
    
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Ok(data.to_string());
        }
    }
    
    // Maybe it's plain JSON (fallback)
    if text.trim().starts_with('{') {
        return Ok(text.trim().to_string());
    }
    
    Err(McpError::Transport(format!("Cannot parse SSE response: {}", text)))
}

#[cfg(feature = "reqwest")]
#[async_trait]
impl McpTransport for HttpTransport {
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        // Use base_url directly (don't append /message)
        let mut req = self.client
            .post(&self.base_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(request);

        // Add custom headers (e.g., Authorization)
        for (key, value) in &self.custom_headers {
            req = req.header(key, value);
        }

        // Add session ID if we have one
        let session_id = self.session_id.lock().await;
        if let Some(ref id) = *session_id {
            req = req.header("X-Session-Id", id);
        }
        drop(session_id);

        let response = req
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        // Store session ID from response
        if let Some(id) = response.headers().get("X-Session-Id") {
            if let Ok(id_str) = id.to_str() {
                *self.session_id.lock().await = Some(id_str.to_string());
            }
        }

        let status = response.status();

        if !status.is_success() {
            return Err(McpError::Http(status.as_u16()));
        }

        // Get response text
        let text = response.text().await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        
        // Parse SSE format
        let json_str = parse_sse_response(&text)?;
        
        let json_response: JsonRpcResponse = serde_json::from_str(&json_str)
            .map_err(|e| McpError::Transport(format!("Failed to parse response: {}", e)))?;

        Ok(json_response)
    }

    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()> {
        // Use base_url directly (don't append /message)
        let mut req = self.client
            .post(&self.base_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(notification);

        // Add custom headers (e.g., Authorization)
        for (key, value) in &self.custom_headers {
            req = req.header(key, value);
        }

        let session_id = self.session_id.lock().await;
        if let Some(ref id) = *session_id {
            req = req.header("X-Session-Id", id);
        }
        drop(session_id);

        let response = req
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        let status = response.status();

        if !status.is_success() {
            return Err(McpError::Http(status.as_u16()));
        }

        Ok(())
    }

    async fn close(&self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Memory transport for testing.
pub struct MemoryTransport {
    responses: Arc<Mutex<std::collections::VecDeque<JsonRpcResponse>>>,
    requests: Arc<Mutex<Vec<JsonRpcRequest>>>,
    notifications: Arc<Mutex<Vec<JsonRpcNotification>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for MemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTransport {
    /// Create a new memory transport.
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            requests: Arc::new(Mutex::new(Vec::new())),
            notifications: Arc::new(Mutex::new(Vec::new())),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Add a response to be returned.
    pub async fn push_response(&self, response: JsonRpcResponse) {
        self.responses.lock().await.push_back(response);
    }

    /// Get recorded requests.
    pub async fn get_requests(&self) -> Vec<JsonRpcRequest> {
        self.requests.lock().await.clone()
    }

    /// Get recorded notifications.
    pub async fn get_notifications(&self) -> Vec<JsonRpcNotification> {
        self.notifications.lock().await.clone()
    }

    /// Clear all recorded data.
    pub async fn clear(&self) {
        self.responses.lock().await.clear();
        self.requests.lock().await.clear();
        self.notifications.lock().await.clear();
    }
}

#[async_trait]
impl McpTransport for MemoryTransport {
    async fn request(&self, request: &JsonRpcRequest) -> McpResult<JsonRpcResponse> {
        self.requests.lock().await.push(request.clone());

        self.responses
            .lock()
            .await
            .pop_front()
            .ok_or(McpError::NoResult)
    }

    async fn notify(&self, notification: &JsonRpcNotification) -> McpResult<()> {
        self.notifications.lock().await.push(notification.clone());
        Ok(())
    }

    async fn close(&self) -> McpResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_transport() {
        let transport = MemoryTransport::new();

        // Push a response
        let response = JsonRpcResponse::success(1, "test");
        transport.push_response(response).await;

        // Make request
        let request = JsonRpcRequest::new(1, "test");
        let resp = transport.request(&request).await.unwrap();

        assert!(!resp.is_error());

        // Check recorded requests
        let requests = transport.get_requests().await;
        assert_eq!(requests.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_transport_notification() {
        let transport = MemoryTransport::new();

        let notification = JsonRpcNotification::new("test/notify");
        transport.notify(&notification).await.unwrap();

        let notifications = transport.get_notifications().await;
        assert_eq!(notifications.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_transport_close() {
        let transport = MemoryTransport::new();
        assert!(transport.is_connected());

        transport.close().await.unwrap();
        assert!(!transport.is_connected());
    }

    #[tokio::test]
    async fn test_spawn_with_env_empty_map() {
        // spawn_with_env with empty HashMap should work like spawn
        let _result = StdioTransport::spawn_with_env("echo", &["hello"], HashMap::new()).await;
        // May fail if echo not available, that's ok for this test
        // Just verify we don't panic
    }

    #[tokio::test]
    async fn test_spawn_with_env_sets_variables() {
        // Test that env vars are passed to child process
        // Use 'printenv' or 'env' command to verify
        let mut env = HashMap::new();
        env.insert("TEST_MCP_VAR".to_string(), "test_value_123".to_string());
        
        // Note: Full test would need to capture output, 
        // but we can at least verify the call doesn't panic
        let _ = StdioTransport::spawn_with_env("echo", &["test"], env).await;
    }
}
