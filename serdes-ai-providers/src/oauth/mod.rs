//! OAuth utilities for PKCE-based authentication flows.
//!
//! This module provides reusable components for OAuth 2.0 PKCE flows:
//!
//! - [`OAuthConfig`]: Configuration for an OAuth provider
//! - [`OAuthContext`]: Runtime state for an in-progress flow (state, verifier, challenge)
//! - [`TokenResponse`]: Tokens returned from the authorization server
//! - [`run_pkce_flow`]: Runs the complete OAuth flow with local callback server
//! - [`refresh_token`]: Refreshes an expired access token
//!
//! Note: This module does NOT handle token storage - that's the application's responsibility.

pub mod config;
mod context;
mod flow;
mod server;

pub use config::OAuthConfig;
pub use context::OAuthContext;
pub use flow::{refresh_token, run_pkce_flow, OAuthError};
pub use server::CallbackServer;

/// Token response from OAuth token endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub id_token: Option<String>,
    #[serde(default)]
    pub token_type: Option<String>,
    #[serde(default)]
    pub expires_in: Option<u64>,
    #[serde(default)]
    pub scope: Option<String>,
}
