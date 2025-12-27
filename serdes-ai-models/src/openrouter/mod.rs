//! OpenRouter model - OpenAI-compatible API routing to multiple providers.

pub mod model;
pub mod types;

pub use model::OpenRouterModel;
pub use types::{DataCollection, OpenRouterExtras, ProviderPreferences, Quantization, models};

/// Create a new OpenRouter model.
pub fn chat(model_name: impl Into<String>, api_key: impl Into<String>) -> OpenRouterModel {
    OpenRouterModel::new(model_name, api_key)
}
