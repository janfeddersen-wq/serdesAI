//! Utility functions for proc macro implementations.

use proc_macro2::TokenStream;
use syn::{Type, GenericArgument, PathArguments};

/// Extract the inner type from an Option<T>
pub fn extract_option_inner(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Option" {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}

/// Check if a type is Option<T>
pub fn is_option_type(ty: &Type) -> bool {
    extract_option_inner(ty).is_some()
}

/// Convert a Rust type to JSON Schema type string
pub fn rust_type_to_json_schema_type(ty: &Type) -> &'static str {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return match segment.ident.to_string().as_str() {
                "String" | "str" => "string",
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => "integer",
                "f32" | "f64" => "number",
                "bool" => "boolean",
                "Vec" => "array",
                _ => "object",
            };
        }
    }
    "object"
}
