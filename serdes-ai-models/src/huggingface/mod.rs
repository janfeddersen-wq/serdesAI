//! HuggingFace Inference API support.
//!
//! This module provides support for the [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
//! and self-hosted [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) endpoints.
//!
//! ## Example
//!
//! ```ignore
//! use serdes_ai_models::huggingface::HuggingFaceModel;
//!
//! // Using the HuggingFace Inference API
//! let model = HuggingFaceModel::from_env("meta-llama/Llama-3.1-8B-Instruct")?;
//!
//! // Or with explicit token
//! let model = HuggingFaceModel::new("mistralai/Mistral-7B-Instruct-v0.2", api_token);
//!
//! // Self-hosted TGI endpoint
//! let model = HuggingFaceModel::new("my-model", api_token)
//!     .with_endpoint("http://localhost:8080/generate");
//! ```
//!
//! ## Environment Variables
//!
//! - `HF_TOKEN` or `HUGGINGFACE_API_TOKEN`: API token for authentication
//!
//! ## Supported Models
//!
//! Any model hosted on HuggingFace Hub that supports the text-generation inference API:
//! - `meta-llama/Llama-3.1-8B-Instruct`
//! - `mistralai/Mistral-7B-Instruct-v0.2`
//! - `google/gemma-2-9b-it`
//! - And many more...

pub mod model;
pub mod types;

pub use model::HuggingFaceModel;
pub use types::{GenerateParameters, GenerateRequest, GenerateResponse};
