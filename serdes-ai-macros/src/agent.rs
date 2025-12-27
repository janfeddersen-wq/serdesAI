//! Agent derive macro implementation.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemStruct};

/// Implementation for `#[agent]` attribute macro
pub fn agent_attribute_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_str = attr.to_string();
    let input = parse_macro_input!(item as ItemStruct);
    let name = &input.ident;
    let vis = &input.vis;
    let fields = &input.fields;
    let attrs = &input.attrs;

    // Parse agent attributes
    let mut model = "openai:gpt-4".to_string();
    let mut system_prompt = String::new();

    // Simple parsing of attr string
    for part in attr_str.split(',') {
        let part = part.trim();
        if part.starts_with("model") {
            if let Some(val) = part.split('=').nth(1) {
                model = val.trim().trim_matches('"').to_string();
            }
        } else if part.starts_with("system_prompt") {
            if let Some(val) = part.split('=').nth(1) {
                system_prompt = val.trim().trim_matches('"').to_string();
            }
        }
    }

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name #fields

        impl #name {
            /// Get the default model name.
            pub fn default_model() -> &'static str {
                #model
            }

            /// Get the system prompt.
            pub fn system_prompt() -> &'static str {
                #system_prompt
            }
        }
    };

    TokenStream::from(expanded)
}

/// Implementation for `#[derive(Agent)]`
pub fn derive_agent_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    // Extract agent attributes
    let mut model = String::new();
    let mut system_prompt = String::new();
    let mut result_type = quote!(String);
    let mut deps_type = quote!(());

    for attr in &input.attrs {
        if attr.path().is_ident("agent") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("model") {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    model = lit.value();
                } else if meta.path.is_ident("system_prompt") {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    system_prompt = lit.value();
                } else if meta.path.is_ident("result") {
                    let value = meta.value()?;
                    let ty: syn::Type = value.parse()?;
                    result_type = quote!(#ty);
                } else if meta.path.is_ident("deps") {
                    let value = meta.value()?;
                    let ty: syn::Type = value.parse()?;
                    deps_type = quote!(#ty);
                }
                Ok(())
            });
        }
    }

    let expanded = quote! {
        impl ::serdes_ai_agent::AgentConfig for #name {
            type Result = #result_type;
            type Deps = #deps_type;

            fn model_name(&self) -> &str {
                #model
            }

            fn system_prompt(&self) -> Option<&str> {
                if #system_prompt.is_empty() {
                    None
                } else {
                    Some(#system_prompt)
                }
            }
        }
    };

    TokenStream::from(expanded)
}
