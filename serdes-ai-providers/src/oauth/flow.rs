//! OAuth PKCE flow execution.

use reqwest::Client;
use std::time::Duration;

use super::{CallbackServer, OAuthConfig, OAuthContext, TokenResponse};
use super::server::CallbackError;

/// Errors that can occur during OAuth flow.
#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
    #[error("Failed to start callback server: {0}")]
    ServerStart(#[from] std::io::Error),
    #[error("Callback error: {0}")]
    Callback(#[from] CallbackError),
    #[error("State mismatch: expected {expected}, got {actual}")]
    StateMismatch { expected: String, actual: String },
    #[error("Token exchange failed: {0}")]
    TokenExchange(String),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

/// Run the complete OAuth PKCE flow.
///
/// This function:
/// 1. Creates a PKCE context (state, verifier, challenge)
/// 2. Starts a local callback server
/// 3. Returns the authorization URL for the user to open
/// 4. Waits for the callback with the authorization code
/// 5. Exchanges the code for tokens
///
/// **Important**: This function does NOT store tokens. The caller is responsible
/// for persisting the returned tokens.
///
/// Returns a tuple of (authorization_url, future that resolves to tokens).
pub async fn run_pkce_flow(config: &OAuthConfig) -> Result<(String, OAuthFlowHandle), OAuthError> {
    let context = OAuthContext::new();
    let server = CallbackServer::start(config)?;
    
    let redirect_uri = config.redirect_uri(server.port());
    let context = context.with_redirect_uri(redirect_uri.clone());
    
    let auth_url = build_authorization_url(config, &context);
    
    let handle = OAuthFlowHandle {
        server,
        context,
        config: config.clone(),
    };
    
    Ok((auth_url, handle))
}

/// Handle to a running OAuth flow.
pub struct OAuthFlowHandle {
    server: CallbackServer,
    context: OAuthContext,
    config: OAuthConfig,
}

impl OAuthFlowHandle {
    /// Get the port the callback server is listening on.
    pub fn port(&self) -> u16 {
        self.server.port()
    }

    /// Get the redirect URI.
    pub fn redirect_uri(&self) -> &str {
        self.context.redirect_uri.as_deref().unwrap_or("")
    }

    /// Wait for the callback and exchange the code for tokens.
    pub async fn wait_for_tokens(self) -> Result<TokenResponse, OAuthError> {
        let timeout = Duration::from_secs(self.config.callback_timeout_secs);
        let result = self.server.wait_for_callback(timeout).await?;
        
        // Verify state
        if result.state != self.context.state {
            return Err(OAuthError::StateMismatch {
                expected: self.context.state,
                actual: result.state,
            });
        }
        
        // Exchange code for tokens
        exchange_code_for_tokens(&self.config, &self.context, &result.code).await
    }
}

/// Build the authorization URL with PKCE parameters.
fn build_authorization_url(config: &OAuthConfig, context: &OAuthContext) -> String {
    let redirect_uri = context.redirect_uri.as_deref().unwrap_or("");
    
    let mut params = vec![
        ("response_type", "code".to_string()),
        ("client_id", config.client_id.clone()),
        ("redirect_uri", redirect_uri.to_string()),
        ("scope", config.scopes.clone()),
        ("code_challenge", context.code_challenge.clone()),
        ("code_challenge_method", "S256".to_string()),
        ("state", context.state.clone()),
    ];
    
    // Claude Code requires "code=true" parameter
    if config.token_url.contains("anthropic.com") {
        params.push(("code", "true".to_string()));
    }
    
    let query = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    
    format!("{}?{}", config.auth_url, query)
}

/// Exchange authorization code for tokens.
async fn exchange_code_for_tokens(
    config: &OAuthConfig,
    context: &OAuthContext,
    code: &str,
) -> Result<TokenResponse, OAuthError> {
    let redirect_uri = context.redirect_uri.as_deref().unwrap_or("");
    let client = Client::new();
    
    // Check if this is Claude/Anthropic (uses JSON) or others (use form-urlencoded)
    let is_anthropic = config.token_url.contains("anthropic.com");
    
    let response = if is_anthropic {
        // Anthropic/Claude uses JSON body with special headers
        let payload = serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "code": code,
            "state": context.state,
            "code_verifier": context.code_verifier,
            "redirect_uri": redirect_uri,
        });
        
        client
            .post(&config.token_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("anthropic-beta", "oauth-2025-04-20")
            .json(&payload)
            .send()
            .await?
    } else {
        // Standard OAuth uses form-urlencoded
        let params = [
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", &config.client_id),
            ("code_verifier", &context.code_verifier),
        ];
        
        client
            .post(&config.token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await?
    };
    
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(OAuthError::TokenExchange(format!(
            "HTTP {}: {}",
            status, body
        )));
    }
    
    let tokens: TokenResponse = response.json().await?;
    Ok(tokens)
}

/// Refresh an expired access token.
///
/// **Important**: This function does NOT store the new tokens. The caller is
/// responsible for persisting them.
pub async fn refresh_token(
    config: &OAuthConfig,
    refresh_token: &str,
) -> Result<TokenResponse, OAuthError> {
    let client = Client::new();
    let is_anthropic = config.token_url.contains("anthropic.com");
    
    let response = if is_anthropic {
        let payload = serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": config.client_id,
        });
        
        client
            .post(&config.token_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("anthropic-beta", "oauth-2025-04-20")
            .json(&payload)
            .send()
            .await?
    } else {
        let params = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", &config.client_id),
        ];
        
        client
            .post(&config.token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await?
    };
    
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(OAuthError::TokenExchange(format!(
            "HTTP {}: {}",
            status, body
        )));
    }
    
    let tokens: TokenResponse = response.json().await?;
    Ok(tokens)
}
