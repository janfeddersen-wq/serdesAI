//! OAuth configuration.

/// Configuration for an OAuth provider.
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    /// OAuth client ID
    pub client_id: String,
    /// Authorization endpoint URL
    pub auth_url: String,
    /// Token endpoint URL  
    pub token_url: String,
    /// OAuth scopes (space-separated)
    pub scopes: String,
    /// Redirect URI host (e.g., "http://localhost")
    pub redirect_host: String,
    /// Redirect URI path (e.g., "auth/callback")
    pub redirect_path: String,
    /// Required port (Some(port) for fixed port, None for dynamic)
    pub required_port: Option<u16>,
    /// Port range for dynamic allocation (start, end inclusive)
    pub port_range: Option<(u16, u16)>,
    /// Callback timeout in seconds
    pub callback_timeout_secs: u64,
}

impl OAuthConfig {
    /// Create a new OAuth configuration.
    pub fn new(
        client_id: impl Into<String>,
        auth_url: impl Into<String>,
        token_url: impl Into<String>,
    ) -> Self {
        Self {
            client_id: client_id.into(),
            auth_url: auth_url.into(),
            token_url: token_url.into(),
            scopes: String::new(),
            redirect_host: "http://localhost".to_string(),
            redirect_path: "callback".to_string(),
            required_port: None,
            port_range: Some((8765, 8795)),
            callback_timeout_secs: 120,
        }
    }

    /// Set OAuth scopes.
    #[must_use]
    pub fn with_scopes(mut self, scopes: impl Into<String>) -> Self {
        self.scopes = scopes.into();
        self
    }

    /// Set required port (fixed port for callback).
    #[must_use]
    pub fn with_required_port(mut self, port: u16) -> Self {
        self.required_port = Some(port);
        self.port_range = None;
        self
    }

    /// Set port range for dynamic allocation.
    #[must_use]
    pub fn with_port_range(mut self, start: u16, end: u16) -> Self {
        self.port_range = Some((start, end));
        self.required_port = None;
        self
    }

    /// Set callback timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.callback_timeout_secs = timeout_secs;
        self
    }

    /// Set redirect path.
    #[must_use]
    pub fn with_redirect_path(mut self, path: impl Into<String>) -> Self {
        self.redirect_path = path.into();
        self
    }

    /// Build the redirect URI for a given port.
    pub fn redirect_uri(&self, port: u16) -> String {
        let host = self.redirect_host.trim_end_matches('/');
        let path = self.redirect_path.trim_start_matches('/');
        format!("{}:{}/{}", host, port, path)
    }
}

/// ChatGPT OAuth configuration (OpenAI Codex).
pub fn chatgpt_oauth_config() -> OAuthConfig {
    OAuthConfig::new(
        "app_EMoamEEZ73f0CkXaXp7hrann",
        "https://auth.openai.com/oauth/authorize",
        "https://auth.openai.com/oauth/token",
    )
    .with_scopes("openid profile email offline_access")
    .with_required_port(1455)
    .with_redirect_path("auth/callback")
    .with_timeout(120)
}

/// Claude Code OAuth configuration.
pub fn claude_code_oauth_config() -> OAuthConfig {
    OAuthConfig::new(
        "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        "https://claude.ai/oauth/authorize",
        "https://console.anthropic.com/v1/oauth/token",
    )
    .with_scopes("org:create_api_key user:profile user:inference")
    .with_port_range(8765, 8795)
    .with_redirect_path("callback")
    .with_timeout(180)
}
