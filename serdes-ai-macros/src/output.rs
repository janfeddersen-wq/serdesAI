//! Output schema derive macro implementation.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Type};

/// Implementation for `#[derive(OutputSchema)]`
pub fn derive_output_schema_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract schema attributes
    let mut description = String::new();
    let mut strict = true;

    for attr in &input.attrs {
        if attr.path().is_ident("output") {
            if let Err(err) = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    description = lit.value();
                } else if meta.path.is_ident("strict") {
                    let value = meta.value()?;
                    let lit: syn::LitBool = value.parse()?;
                    strict = lit.value;
                } else {
                    return Err(meta.error("unknown `output` attribute"));
                }
                Ok(())
            }) {
                return err.to_compile_error().into();
            }
        }
    }

    // Generate JSON schema from struct
    let schema_impl = generate_json_schema(&input);

    let expanded = quote! {
        impl #impl_generics ::serdes_ai_output::OutputSchema for #name #ty_generics #where_clause {
            fn json_schema() -> ::serde_json::Value {
                #schema_impl
            }

            fn strict() -> bool {
                #strict
            }

            fn description() -> Option<&'static str> {
                if #description.is_empty() {
                    None
                } else {
                    Some(#description)
                }
            }

            fn validate(_output: &str) -> ::serdes_ai_output::ValidationResult<Self>
            where
                Self: Sized + ::serde::de::DeserializeOwned,
            {
                match ::serde_json::from_str(_output) {
                    Ok(value) => ::serdes_ai_output::ValidationResult::Valid(value),
                    Err(e) => ::serdes_ai_output::ValidationResult::Invalid(e.to_string()),
                }
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_json_schema(input: &DeriveInput) -> TokenStream2 {
    let name = input.ident.to_string();

    match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let field_entries: Vec<TokenStream2> = fields
                    .named
                    .iter()
                    .map(|f| {
                        let field_name = f.ident.as_ref().unwrap().to_string();
                        let ty = &f.ty;
                        let type_schema = type_to_schema(ty);

                        // Extract field description from doc comments
                        let description = f
                            .attrs
                            .iter()
                            .filter(|a| a.path().is_ident("doc"))
                            .filter_map(|a| {
                                if let syn::Meta::NameValue(nv) = &a.meta {
                                    if let syn::Expr::Lit(lit) = &nv.value {
                                        if let syn::Lit::Str(s) = &lit.lit {
                                            return Some(s.value().trim().to_string());
                                        }
                                    }
                                }
                                None
                            })
                            .collect::<Vec<_>>()
                            .join(" ");

                        if description.is_empty() {
                            quote! {
                                (#field_name, #type_schema)
                            }
                        } else {
                            quote! {
                                (#field_name, {
                                    let mut schema = #type_schema;
                                    if let Some(obj) = schema.as_object_mut() {
                                        obj.insert("description".to_string(), ::serde_json::json!(#description));
                                    }
                                    schema
                                })
                            }
                        }
                    })
                    .collect();

                let required_fields: Vec<TokenStream2> = fields
                    .named
                    .iter()
                    .filter(|f| !is_option_type(&f.ty))
                    .map(|f| {
                        let name = f.ident.as_ref().unwrap().to_string();
                        quote!(#name)
                    })
                    .collect();

                quote! {
                    {
                        let properties: ::std::collections::HashMap<&str, ::serde_json::Value> = [
                            #(#field_entries),*
                        ].into_iter().collect();

                        let required: Vec<&str> = vec![#(#required_fields),*];

                        ::serde_json::json!({
                            "type": "object",
                            "title": #name,
                            "properties": properties,
                            "required": required,
                            "additionalProperties": false
                        })
                    }
                }
            }
            syn::Fields::Unnamed(_) => {
                quote!(::serde_json::json!({ "type": "array" }))
            }
            syn::Fields::Unit => {
                quote!(::serde_json::json!({ "type": "null" }))
            }
        },
        syn::Data::Enum(data) => {
            let variants: Vec<String> = data.variants.iter().map(|v| v.ident.to_string()).collect();

            quote! {
                ::serde_json::json!({
                    "type": "string",
                    "enum": [#(#variants),*]
                })
            }
        }
        syn::Data::Union(_) => {
            quote!(::serde_json::json!({ "type": "object" }))
        }
    }
}

fn type_to_schema(ty: &Type) -> TokenStream2 {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let ident = seg.ident.to_string();
            return match ident.as_str() {
                "String" | "str" => quote!(::serde_json::json!({ "type": "string" })),
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
                    quote!(::serde_json::json!({ "type": "integer" }))
                }
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => {
                    quote!(::serde_json::json!({ "type": "integer" }))
                }
                "f32" | "f64" => quote!(::serde_json::json!({ "type": "number" })),
                "bool" => quote!(::serde_json::json!({ "type": "boolean" })),
                "Vec" => {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            let inner_schema = type_to_schema(inner);
                            return quote! {
                                ::serde_json::json!({
                                    "type": "array",
                                    "items": #inner_schema
                                })
                            };
                        }
                    }
                    quote!(::serde_json::json!({ "type": "array" }))
                }
                "Option" => {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            let inner_schema = type_to_schema(inner);
                            return quote! {
                                {
                                    let mut schema = #inner_schema;
                                    if let Some(obj) = schema.as_object_mut() {
                                        // Make it nullable
                                        if let Some(ty) = obj.get("type") {
                                            obj.insert(
                                                "type".to_string(),
                                                ::serde_json::json!([ty, "null"])
                                            );
                                        }
                                    }
                                    schema
                                }
                            };
                        }
                    }
                    quote!(::serde_json::json!({ "type": ["string", "null"] }))
                }
                "HashMap" | "BTreeMap" => {
                    quote!(::serde_json::json!({
                        "type": "object",
                        "additionalProperties": true
                    }))
                }
                _ => quote!(::serde_json::json!({ "type": "object" })),
            };
        }
    }
    quote!(::serde_json::json!({ "type": "string" }))
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            return seg.ident == "Option";
        }
    }
    false
}
