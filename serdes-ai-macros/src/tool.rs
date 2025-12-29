//! Tool derive macro and attribute implementation.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::{
    parse_macro_input, punctuated::Punctuated, DeriveInput, Error, FnArg, ItemFn, Meta, Pat, Token,
    Type,
};

struct ParamInfo {
    ident: syn::Ident,
    ty: Type,
}

/// Implementation for `#[derive(Tool)]`
pub fn derive_tool_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract tool attributes
    let mut description = String::new();
    let mut tool_name = to_snake_case(&name_str);
    let mut strict = false;

    for attr in &input.attrs {
        if attr.path().is_ident("tool") {
            if let Err(err) = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    tool_name = lit.value();
                } else if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    description = lit.value();
                } else if meta.path.is_ident("strict") {
                    if !meta.input.is_empty() {
                        return Err(meta.error("`strict` does not take a value"));
                    }
                    strict = true;
                } else {
                    return Err(meta.error("unknown `tool` attribute"));
                }
                Ok(())
            }) {
                return err.to_compile_error().into();
            }
        }
    }

    // Generate schema from struct fields
    let schema_fields = generate_struct_schema(&input);

    let expanded = quote! {
        impl #impl_generics ::serdes_ai_tools::Tool for #name #ty_generics #where_clause {
            fn definition(&self) -> ::serdes_ai_tools::ToolDefinition {
                ::serdes_ai_tools::ToolDefinition {
                    name: #tool_name.to_string(),
                    description: #description.to_string(),
                    parameters_json_schema: {
                        #schema_fields
                    },
                    strict: Some(#strict),
                    outer_typed_dict_key: None,
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Implementation for `#[tool]` attribute macro
pub fn tool_attribute_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();
    let fn_block = &input.block;
    let fn_inputs = &input.sig.inputs;
    let fn_output = &input.sig.output;
    let fn_vis = &input.vis;
    let fn_asyncness = &input.sig.asyncness;

    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    let args = match parser.parse(attr) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };

    let mut description = String::new();
    let mut tool_name = fn_name_str.clone();
    let mut strict = false;

    for meta in args {
        match meta {
            Meta::NameValue(nv) if nv.path.is_ident("name") => {
                match parse_lit_str(&nv.value) {
                    Ok(value) => tool_name = value,
                    Err(err) => return err.to_compile_error().into(),
                }
            }
            Meta::NameValue(nv) if nv.path.is_ident("description") => {
                match parse_lit_str(&nv.value) {
                    Ok(value) => description = value,
                    Err(err) => return err.to_compile_error().into(),
                }
            }
            Meta::Path(path) if path.is_ident("strict") => {
                strict = true;
            }
            Meta::NameValue(nv) if nv.path.is_ident("strict") => {
                match parse_lit_bool(&nv.value) {
                    Ok(value) => strict = value,
                    Err(err) => return err.to_compile_error().into(),
                }
            }
            _ => {
                return Error::new_spanned(meta, "unknown `tool` attribute")
                    .to_compile_error()
                    .into();
            }
        }
    }

    // Extract parameters (skip self or RunContext)
    let params: Vec<ParamInfo> = match input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pat_type) => Some(pat_type),
            _ => None,
        })
        .skip(1)
        .map(|pat_type| {
            let ident = extract_pat_ident(&pat_type.pat)?;
            Ok(ParamInfo {
                ident,
                ty: (*pat_type.ty).clone(),
            })
        })
        .collect::<Result<_, Error>>()
    {
        Ok(params) => params,
        Err(err) => return err.to_compile_error().into(),
    };

    // Generate struct for parameters
    let struct_name = format_ident!("{}Args", to_pascal_case(&fn_name_str));
    let tool_struct_name = format_ident!("{}Tool", to_pascal_case(&fn_name_str));

    let param_fields: Vec<TokenStream2> = params
        .iter()
        .map(|param| {
            let ident = &param.ident;
            let ty = &param.ty;
            quote! { pub #ident: #ty }
        })
        .collect();

    let param_schema = generate_param_schema(&params);

    let call_args: Vec<TokenStream2> = params
        .iter()
        .map(|param| {
            let ident = &param.ident;
            quote!(args.#ident)
        })
        .collect();

    let call_expr = if fn_asyncness.is_some() {
        quote!(#fn_name(ctx, #(#call_args),*).await)
    } else {
        quote!(#fn_name(ctx, #(#call_args),*))
    };

    let expanded = quote! {
        /// Auto-generated arguments struct
        #[derive(Debug, Clone, ::serde::Serialize, ::serde::Deserialize)]
        pub struct #struct_name {
            #(#param_fields),*
        }

        /// Auto-generated tool wrapper
        #[derive(Debug, Clone, Default)]
        pub struct #tool_struct_name;

        impl #tool_struct_name {
            /// Create a new instance.
            pub fn new() -> Self {
                Self
            }
        }

        #[::async_trait::async_trait]
        impl<Deps> ::serdes_ai_tools::Tool<Deps> for #tool_struct_name
        where
            Deps: Send + Sync + 'static,
        {
            fn definition(&self) -> ::serdes_ai_tools::ToolDefinition {
                let mut def = ::serdes_ai_tools::ToolDefinition::new(#tool_name, #description)
                    .with_parameters(#param_schema);
                if #strict {
                    def = def.with_strict(true);
                }
                def
            }

            async fn call(
                &self,
                ctx: &::serdes_ai_tools::RunContext<Deps>,
                args: ::serde_json::Value,
            ) -> ::serdes_ai_tools::ToolResult {
                let args: #struct_name = ::serde_json::from_value(args)
                    .map_err(|e| ::serdes_ai_tools::ToolError::invalid_args(e.to_string()))?;
                #call_expr
            }
        }

        // Keep the original function
        #fn_vis #fn_asyncness fn #fn_name(#fn_inputs) #fn_output #fn_block
    };

    TokenStream::from(expanded)
}

fn generate_struct_schema(input: &DeriveInput) -> TokenStream2 {
    match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let field_schemas: Vec<TokenStream2> = fields
                    .named
                    .iter()
                    .map(|f| {
                        let name = f.ident.as_ref().unwrap().to_string();
                        let ty = &f.ty;
                        let type_str = type_to_json_type(ty);
                        let required = !is_option_type(ty);

                        quote! {
                            properties.insert(
                                #name.to_string(),
                                ::serde_json::json!({ "type": #type_str })
                            );
                            if #required {
                                required.push(#name.to_string());
                            }
                        }
                    })
                    .collect();

                quote! {
                    {
                        let mut properties = ::std::collections::HashMap::new();
                        let mut required = Vec::new();
                        #(#field_schemas)*
                        ::serdes_ai_tools::ObjectJsonSchema {
                            properties,
                            required,
                            additional_properties: Some(false),
                            ..Default::default()
                        }
                    }
                }
            }
            _ => quote!(::serdes_ai_tools::ObjectJsonSchema::default()),
        },
        _ => quote!(::serdes_ai_tools::ObjectJsonSchema::default()),
    }
}

fn generate_param_schema(params: &[ParamInfo]) -> TokenStream2 {
    let field_schemas: Vec<TokenStream2> = params
        .iter()
        .map(|param| {
            let name = param.ident.to_string();
            let ty = &param.ty;
            let type_str = type_to_json_type(ty);
            let required = !is_option_type(ty);

            quote! {
                schema.add_property(
                    #name,
                    ::serde_json::json!({ "type": #type_str }),
                    #required,
                );
            }
        })
        .collect();

    quote! {
        {
            let mut schema = ::serdes_ai_tools::ObjectJsonSchema::new();
            #(#field_schemas)*
            schema
        }
    }
}

fn type_to_json_type(ty: &Type) -> &'static str {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let ident = seg.ident.to_string();
            return match ident.as_str() {
                "String" | "str" => "string",
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => "integer",
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => "integer",
                "f32" | "f64" => "number",
                "bool" => "boolean",
                "Vec" => "array",
                "Option" => {
                    // Get inner type
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            return type_to_json_type(inner);
                        }
                    }
                    "string"
                }
                _ => "object",
            };
        }
    }
    "string"
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            return seg.ident == "Option";
        }
    }
    false
}

fn extract_pat_ident(pat: &Pat) -> Result<syn::Ident, Error> {
    match pat {
        Pat::Ident(ident) => Ok(ident.ident.clone()),
        _ => Err(Error::new_spanned(pat, "tool parameters must be identifiers")),
    }
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

fn parse_lit_bool(expr: &syn::Expr) -> Result<bool, Error> {
    match expr {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Bool(value) => Ok(value.value()),
            _ => Err(Error::new_spanned(expr, "expected a boolean literal")),
        },
        _ => Err(Error::new_spanned(expr, "expected a boolean literal")),
    }
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' || c == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_uppercase().next().unwrap());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}
