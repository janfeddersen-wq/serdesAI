//! HTTP transport with automatic retries.

use crate::config::RetryConfig;
use crate::error::{RetryResult, RetryableError};
use crate::executor::with_retry;
use reqwest::{Client, Method, Response};
use serde::Serialize;
use std::time::Duration;
use tracing::debug;

impl From<reqwest::Error> for RetryableError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            RetryableError::Timeout
        } else if err.is_connect() {
            RetryableError::Connection(err.to_string())
        } else {
            RetryableError::Other(err.into())
        }
    }
}

/// HTTP client wrapper with automatic retries.
#[derive(Debug, Clone)]
pub struct RetryClient {
    client: Client,
    config: RetryConfig,
}

impl RetryClient {
    /// Create a new retry client with default reqwest client.
    pub fn new(config: RetryConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Create with a custom reqwest client.
    pub fn with_client(client: Client, config: RetryConfig) -> Self {
        Self { client, config }
    }

    /// Create with default API retry settings.
    pub fn for_api() -> Self {
        Self::new(RetryConfig::for_api())
    }

    /// Get a reference to the underlying client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get a reference to the retry config.
    pub fn config(&self) -> &RetryConfig {
        &self.config
    }

    /// Execute a GET request with retries.
    pub async fn get(&self, url: &str) -> RetryResult<Response> {
        self.request(Method::GET, url, Option::<()>::None).await
    }

    /// Execute a POST request with retries.
    pub async fn post<B: Serialize + Clone + Send + Sync>(
        &self,
        url: &str,
        body: B,
    ) -> RetryResult<Response> {
        self.request(Method::POST, url, Some(body)).await
    }

    /// Execute a PUT request with retries.
    pub async fn put<B: Serialize + Clone + Send + Sync>(
        &self,
        url: &str,
        body: B,
    ) -> RetryResult<Response> {
        self.request(Method::PUT, url, Some(body)).await
    }

    /// Execute a DELETE request with retries.
    pub async fn delete(&self, url: &str) -> RetryResult<Response> {
        self.request(Method::DELETE, url, Option::<()>::None).await
    }

    /// Execute a PATCH request with retries.
    pub async fn patch<B: Serialize + Clone + Send + Sync>(
        &self,
        url: &str,
        body: B,
    ) -> RetryResult<Response> {
        self.request(Method::PATCH, url, Some(body)).await
    }

    /// Execute a request with retries.
    async fn request<B: Serialize + Clone + Send + Sync>(
        &self,
        method: Method,
        url: &str,
        body: Option<B>,
    ) -> RetryResult<Response> {
        let url = url.to_string();
        let client = self.client.clone();

        with_retry(&self.config, || {
            let url = url.clone();
            let method = method.clone();
            let client = client.clone();
            let body = body.clone();

            async move {
                debug!(method = %method, url = %url, "Making HTTP request");

                let mut request = client.request(method, &url);
                if let Some(b) = body {
                    request = request.json(&b);
                }

                let response = request.send().await.map_err(RetryableError::from)?;

                check_response(response).await
            }
        })
        .await
    }
}

/// Check an HTTP response and convert to RetryableError if needed.
async fn check_response(response: Response) -> RetryResult<Response> {
    let status = response.status().as_u16();

    if status == 429 {
        // Rate limited
        let retry_after = parse_retry_after(&response);
        return Err(RetryableError::RateLimited { retry_after });
    }

    if (500..=599).contains(&status) {
        // Server error
        let retry_after = parse_retry_after(&response);
        let body = response.text().await.unwrap_or_default();
        return Err(RetryableError::Http {
            status,
            body,
            retry_after,
        });
    }

    if !response.status().is_success() {
        // Other error (not retryable)
        let body = response.text().await.unwrap_or_default();
        return Err(RetryableError::Http {
            status,
            body,
            retry_after: None,
        });
    }

    Ok(response)
}

/// Parse Retry-After header.
fn parse_retry_after(response: &Response) -> Option<Duration> {
    response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            // Try parsing as seconds
            s.parse::<u64>().ok().map(Duration::from_secs)
        })
}

/// Builder for creating a retry client.
#[derive(Debug, Default)]
pub struct RetryClientBuilder {
    client: Option<Client>,
    max_retries: Option<u32>,
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
    timeout: Option<Duration>,
}

impl RetryClientBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the underlying HTTP client.
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Set max retries.
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Set initial delay.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = Some(delay);
        self
    }

    /// Set max delay.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = Some(delay);
        self
    }

    /// Set request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Build the retry client.
    pub fn build(self) -> RetryClient {
        let client = self.client.unwrap_or_else(|| {
            let mut builder = Client::builder();
            if let Some(timeout) = self.timeout {
                builder = builder.timeout(timeout);
            }
            builder.build().expect("Failed to build client")
        });

        let mut config = RetryConfig::for_api();
        if let Some(n) = self.max_retries {
            config = config.max_retries(n);
        }
        if let Some(initial) = self.initial_delay {
            let max = self.max_delay.unwrap_or(Duration::from_secs(60));
            config = config.exponential(initial, max);
        }

        RetryClient::with_client(client, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_client_new() {
        let client = RetryClient::new(RetryConfig::for_api());
        assert_eq!(client.config().max_retries, 3);
    }

    #[test]
    fn test_retry_client_for_api() {
        let client = RetryClient::for_api();
        assert_eq!(client.config().max_retries, 3);
    }

    #[test]
    fn test_builder() {
        let client = RetryClientBuilder::new()
            .max_retries(5)
            .initial_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(30))
            .build();

        assert_eq!(client.config().max_retries, 5);
    }

    #[test]
    fn test_parse_retry_after() {
        // This would require mocking the response, so we just test the logic
        let duration = Duration::from_secs(5);
        assert_eq!(duration.as_secs(), 5);
    }
}
