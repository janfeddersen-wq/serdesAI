//! OAuth PKCE context.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use sha2::{Digest, Sha256};

/// Runtime state for an in-progress OAuth PKCE flow.
#[derive(Debug, Clone)]
pub struct OAuthContext {
    /// Random state parameter for CSRF protection
    pub state: String,
    /// PKCE code verifier (random string)
    pub code_verifier: String,
    /// PKCE code challenge (SHA256 hash of verifier)
    pub code_challenge: String,
    /// When this context was created (Unix timestamp)
    pub created_at: u64,
    /// Assigned redirect URI (set after server starts)
    pub redirect_uri: Option<String>,
}

impl OAuthContext {
    /// Create a new OAuth context with PKCE parameters.
    pub fn new() -> Self {
        let state = Self::generate_random_string(32);
        let code_verifier = Self::generate_random_string(64);
        let code_challenge = Self::compute_code_challenge(&code_verifier);
        
        Self {
            state,
            code_verifier,
            code_challenge,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            redirect_uri: None,
        }
    }

    /// Generate a random hex string of the given length (in bytes).
    fn generate_random_string(bytes: usize) -> String {
        use std::fmt::Write;
        let mut rng_bytes = vec![0u8; bytes];
        getrandom::getrandom(&mut rng_bytes).expect("Failed to generate random bytes");
        let mut s = String::with_capacity(bytes * 2);
        for b in rng_bytes {
            write!(s, "{:02x}", b).unwrap();
        }
        s
    }

    /// Compute PKCE code challenge from verifier (S256 method).
    fn compute_code_challenge(verifier: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let digest = hasher.finalize();
        URL_SAFE_NO_PAD.encode(digest)
    }

    /// Set the redirect URI.
    pub fn with_redirect_uri(mut self, uri: String) -> Self {
        self.redirect_uri = Some(uri);
        self
    }

    /// Check if this context has expired (default 5 min lifetime).
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now - self.created_at > 300
    }
}

impl Default for OAuthContext {
    fn default() -> Self {
        Self::new()
    }
}
