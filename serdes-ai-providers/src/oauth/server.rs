//! Local HTTP callback server for OAuth redirects.

use std::net::TcpListener;

use super::config::OAuthConfig;

/// Result of the OAuth callback.
#[derive(Debug, Clone)]
pub struct CallbackResult {
    /// Authorization code from the callback
    pub code: String,
    /// State parameter (should match the one sent)
    pub state: String,
}

/// Local HTTP server that listens for OAuth callbacks.
pub struct CallbackServer {
    port: u16,
    listener: TcpListener,
}

impl CallbackServer {
    /// Try to start a callback server on an available port.
    pub fn start(config: &OAuthConfig) -> Result<Self, std::io::Error> {
        if let Some(port) = config.required_port {
            // Fixed port required
            let addr = format!("127.0.0.1:{}", port);
            let listener = TcpListener::bind(&addr)?;
            listener.set_nonblocking(true)?;
            Ok(Self { port, listener })
        } else if let Some((start, end)) = config.port_range {
            // Try ports in range
            for port in start..=end {
                let addr = format!("127.0.0.1:{}", port);
                match TcpListener::bind(&addr) {
                    Ok(listener) => {
                        listener.set_nonblocking(true)?;
                        return Ok(Self { port, listener });
                    }
                    Err(_) => continue,
                }
            }
            Err(std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                format!("No available ports in range {}-{}", start, end),
            ))
        } else {
            // Random port
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            listener.set_nonblocking(true)?;
            Ok(Self { port, listener })
        }
    }

    /// Get the port this server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Wait for the OAuth callback.
    ///
    /// Returns the authorization code and state from the callback URL.
    pub async fn wait_for_callback(
        self,
        timeout: std::time::Duration,
    ) -> Result<CallbackResult, CallbackError> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener as TokioTcpListener;

        let listener = TokioTcpListener::from_std(self.listener)?;

        let result = tokio::time::timeout(timeout, async {
            loop {
                let (mut stream, _) = listener.accept().await?;

                let mut buffer = [0u8; 4096];
                let n = stream.read(&mut buffer).await?;
                let request = String::from_utf8_lossy(&buffer[..n]);

                // Parse the GET request to extract query parameters
                if let Some(result) = Self::parse_callback_request(&request) {
                    // Send success response
                    let response = Self::success_response();
                    let _ = stream.write_all(response.as_bytes()).await;
                    return Ok(result);
                } else if request.contains("GET /") {
                    // Send error response for invalid callback
                    let response = Self::error_response("Missing code or state parameter");
                    let _ = stream.write_all(response.as_bytes()).await;
                }
            }
        })
        .await;

        match result {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(CallbackError::Io(e)),
            Err(_) => Err(CallbackError::Timeout),
        }
    }

    fn parse_callback_request(request: &str) -> Option<CallbackResult> {
        // Parse: GET /callback?code=xxx&state=yyy HTTP/1.1
        let first_line = request.lines().next()?;
        let path = first_line.split_whitespace().nth(1)?;

        let query_start = path.find('?')?;
        let query = &path[query_start + 1..];

        let mut code = None;
        let mut state = None;

        for pair in query.split('&') {
            let mut parts = pair.splitn(2, '=');
            match (parts.next(), parts.next()) {
                (Some("code"), Some(v)) => code = Some(urlencoding::decode(v).ok()?.into_owned()),
                (Some("state"), Some(v)) => state = Some(urlencoding::decode(v).ok()?.into_owned()),
                _ => {}
            }
        }

        Some(CallbackResult {
            code: code?,
            state: state?,
        })
    }

    fn success_response() -> String {
        let body = r#"<!DOCTYPE html>
<html>
<head><title>Authentication Successful</title></head>
<body style="font-family: system-ui; text-align: center; padding: 50px;">
<h1>🎉 Authentication Successful!</h1>
<p>You can close this window and return to the CLI.</p>
</body>
</html>"#;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }

    fn error_response(message: &str) -> String {
        let body = format!(
            r#"<!DOCTYPE html>
<html>
<head><title>Authentication Failed</title></head>
<body style="font-family: system-ui; text-align: center; padding: 50px;">
<h1>❌ Authentication Failed</h1>
<p>{}</p>
</body>
</html>"#,
            message
        );
        format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }
}

/// Errors that can occur during callback handling.
#[derive(Debug, thiserror::Error)]
pub enum CallbackError {
    #[error("Callback timeout")]
    Timeout,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("State mismatch")]
    StateMismatch,
}
