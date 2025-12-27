//! OutputSchema derive macro implementation.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Implementation for `#[derive(OutputSchema)]`
pub fn derive_output_schema_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    let expanded = quote! {
        impl ::serdes_ai_output::OutputSchema for #name {
            fn json_schema() -> ::serde_json::Value {
                ::serde_json::json!({
                    "type": "object",
                    "title": stringify!(#name)
                })
            }
            
            fn schema_name() -> &'static str {
                stringify!(#name)
            }
        }
    };
    
    TokenStream::from(expanded)
}
