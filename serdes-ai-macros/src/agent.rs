//! Agent derive macro implementation.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{parse_macro_input, punctuated::Punctuated, Error, ItemStruct, Meta, Token};

/// Implementation for `#[agent]` attribute macro
pub fn agent_attribute_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let name = &input.ident;
    let vis = &input.vis;
    let fields = &input.fields;
    let attrs = &input.attrs;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    let args = match parser.parse(attr) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };

    // Parse agent attributes
    let mut model = "openai:gpt-4".to_string();
    let mut system_prompt = String::new();

    for meta in args {
        match meta {
            Meta::NameValue(nv) if nv.path.is_ident("model") => match parse_lit_str(&nv.value) {
                Ok(value) => model = value,
                Err(err) => return err.to_compile_error().into(),
            },
            Meta::NameValue(nv) if nv.path.is_ident("system_prompt") => {
                match parse_lit_str(&nv.value) {
                    Ok(value) => system_prompt = value,
                    Err(err) => return err.to_compile_error().into(),
                }
            }
            _ => {
                return Error::new_spanned(meta, "unknown `agent` attribute")
                    .to_compile_error()
                    .into();
            }
        }
    }

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name #generics #fields

        impl #impl_generics #name #ty_generics #where_clause {
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
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract agent attributes
    let mut model = String::new();
    let mut system_prompt = String::new();
    let mut result_type = quote!(String);
    let mut deps_type = quote!(());

    for attr in &input.attrs {
        if attr.path().is_ident("agent") {
            if let Err(err) = attr.parse_nested_meta(|meta| {
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
                } else {
                    return Err(meta.error("unknown `agent` attribute"));
                }
                Ok(())
            }) {
                return err.to_compile_error().into();
            }
        }
    }

    let expanded = quote! {
        impl #impl_generics ::serdes_ai_agent::AgentConfig for #name #ty_generics #where_clause {
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

fn parse_lit_str(expr: &syn::Expr) -> Result<String, Error> {
    match expr {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Str(value) => Ok(value.value()),
            _ => Err(Error::new_spanned(expr, "expected a string literal")),
        },
        _ => Err(Error::new_spanned(expr, "expected a string literal")),
    }
}
