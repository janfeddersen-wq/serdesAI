//! Image generation tool for AI-powered image creation.
//!
//! This module provides a configurable image generation tool that can be
//! integrated with various image generation providers (DALL-E, Stable Diffusion, etc.).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{
    definition::{ObjectJsonSchema, ToolDefinition},
    errors::ToolError,
    return_types::{ToolResult, ToolReturn},
    schema::SchemaBuilder,
    tool::Tool,
    RunContext,
};

// ============================================================================
// Enums
// ============================================================================

/// Background style for generated images.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageBackground {
    /// Transparent background (where supported).
    Transparent,
    /// Opaque/solid background.
    Opaque,
    /// Let the model decide.
    #[default]
    Auto,
}

impl std::fmt::Display for ImageBackground {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transparent => write!(f, "transparent"),
            Self::Opaque => write!(f, "opaque"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Output format for generated images.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// PNG format (lossless, supports transparency).
    #[default]
    Png,
    /// WebP format (modern, efficient).
    Webp,
    /// JPEG format (lossy, smaller file size).
    Jpeg,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Png => write!(f, "png"),
            Self::Webp => write!(f, "webp"),
            Self::Jpeg => write!(f, "jpeg"),
        }
    }
}

/// Quality level for generated images.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageQuality {
    /// Low quality (faster, cheaper).
    Low,
    /// Medium quality.
    Medium,
    /// High quality (slower, more expensive).
    High,
    /// Let the provider decide.
    #[default]
    Auto,
}

impl std::fmt::Display for ImageQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Aspect ratio for generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageAspectRatio {
    /// Ultra-wide 21:9 ratio.
    #[serde(rename = "21_9")]
    R21_9,
    /// Wide 16:9 ratio (HD video).
    #[serde(rename = "16_9")]
    R16_9,
    /// Classic 3:2 ratio (photography).
    #[serde(rename = "3_2")]
    R3_2,
    /// Standard 4:3 ratio.
    #[serde(rename = "4_3")]
    R4_3,
    /// Square 1:1 ratio.
    #[serde(rename = "1_1")]
    R1_1,
    /// Portrait 3:4 ratio.
    #[serde(rename = "3_4")]
    R3_4,
    /// Portrait 2:3 ratio.
    #[serde(rename = "2_3")]
    R2_3,
    /// Tall portrait 9:16 ratio.
    #[serde(rename = "9_16")]
    R9_16,
    /// Ultra-tall 9:21 ratio.
    #[serde(rename = "9_21")]
    R9_21,
}

impl Default for ImageAspectRatio {
    fn default() -> Self {
        Self::R1_1
    }
}

impl std::fmt::Display for ImageAspectRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::R21_9 => write!(f, "21:9"),
            Self::R16_9 => write!(f, "16:9"),
            Self::R3_2 => write!(f, "3:2"),
            Self::R4_3 => write!(f, "4:3"),
            Self::R1_1 => write!(f, "1:1"),
            Self::R3_4 => write!(f, "3:4"),
            Self::R2_3 => write!(f, "2:3"),
            Self::R9_16 => write!(f, "9:16"),
            Self::R9_21 => write!(f, "9:21"),
        }
    }
}

/// Predefined image sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSize {
    /// Let the provider decide based on aspect ratio.
    Auto,
    /// 256x256 pixels (small).
    #[serde(rename = "256x256")]
    Size256x256,
    /// 512x512 pixels (medium).
    #[serde(rename = "512x512")]
    Size512x512,
    /// 1024x1024 pixels (standard).
    #[serde(rename = "1024x1024")]
    Size1024x1024,
    /// 1024x1792 pixels (portrait).
    #[serde(rename = "1024x1792")]
    Size1024x1792,
    /// 1792x1024 pixels (landscape).
    #[serde(rename = "1792x1024")]
    Size1792x1024,
    /// 2048x2048 pixels (large).
    #[serde(rename = "2048x2048")]
    Size2048x2048,
}

impl Default for ImageSize {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Size256x256 => write!(f, "256x256"),
            Self::Size512x512 => write!(f, "512x512"),
            Self::Size1024x1024 => write!(f, "1024x1024"),
            Self::Size1024x1792 => write!(f, "1024x1792"),
            Self::Size1792x1024 => write!(f, "1792x1024"),
            Self::Size2048x2048 => write!(f, "2048x2048"),
        }
    }
}

// ============================================================================
// Main Tool Struct
// ============================================================================

/// Image generation tool for creating images from text prompts.
///
/// This tool allows agents to generate images using AI models.
/// It requires integration with an external image generation provider.
///
/// # Example
///
/// ```ignore
/// use serdes_ai_tools::builtin::ImageGenerationTool;
///
/// let tool = ImageGenerationTool::new()
///     .quality(ImageQuality::High)
///     .output_format(OutputFormat::Png)
///     .size(ImageSize::Size1024x1024);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationTool {
    /// Background style for the image.
    pub background: ImageBackground,
    /// Output format for the generated image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
    /// Quality level for generation.
    pub quality: ImageQuality,
    /// Aspect ratio for the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<ImageAspectRatio>,
    /// Predefined size for the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<ImageSize>,
    /// Output compression level (0-100, where applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    /// Number of partial images for streaming (0-3).
    pub partial_images: u8,
}

impl Default for ImageGenerationTool {
    fn default() -> Self {
        Self {
            background: ImageBackground::Auto,
            output_format: None,
            quality: ImageQuality::Auto,
            aspect_ratio: None,
            size: None,
            output_compression: None,
            partial_images: 0,
        }
    }
}

impl ImageGenerationTool {
    /// Create a new image generation tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the background style.
    #[must_use]
    pub fn background(mut self, background: ImageBackground) -> Self {
        self.background = background;
        self
    }

    /// Set the output format.
    #[must_use]
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Set the quality level.
    #[must_use]
    pub fn quality(mut self, quality: ImageQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the aspect ratio.
    #[must_use]
    pub fn aspect_ratio(mut self, ratio: ImageAspectRatio) -> Self {
        self.aspect_ratio = Some(ratio);
        self
    }

    /// Set the image size.
    #[must_use]
    pub fn size(mut self, size: ImageSize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set the output compression level (0-100).
    ///
    /// Only applies to lossy formats like JPEG and WebP.
    #[must_use]
    pub fn output_compression(mut self, compression: u8) -> Self {
        self.output_compression = Some(compression.min(100));
        self
    }

    /// Set the number of partial images for streaming.
    ///
    /// Valid values are 0-3. Higher values provide more intermediate
    /// results during generation.
    #[must_use]
    pub fn partial_images(mut self, count: u8) -> Self {
        self.partial_images = count.min(3);
        self
    }

    /// Get the tool schema.
    fn schema() -> ObjectJsonSchema {
        SchemaBuilder::new()
            .string("prompt", "The text prompt describing the image to generate", true)
            .enum_values(
                "background",
                "Background style (transparent, opaque, or auto)",
                &["transparent", "opaque", "auto"],
                false,
            )
            .enum_values(
                "output_format",
                "Output format for the image",
                &["png", "webp", "jpeg"],
                false,
            )
            .enum_values(
                "quality",
                "Quality level for generation",
                &["low", "medium", "high", "auto"],
                false,
            )
            .enum_values(
                "aspect_ratio",
                "Aspect ratio for the image",
                &["21_9", "16_9", "3_2", "4_3", "1_1", "3_4", "2_3", "9_16", "9_21"],
                false,
            )
            .enum_values(
                "size",
                "Predefined size for the image",
                &["auto", "256x256", "512x512", "1024x1024", "1024x1792", "1792x1024", "2048x2048"],
                false,
            )
            .integer_constrained(
                "output_compression",
                "Compression level for lossy formats (0-100)",
                false,
                Some(0),
                Some(100),
            )
            .build()
    }

    /// Generate an image (stub - integrate with actual provider).
    async fn generate(&self, prompt: &str) -> Result<ImageGenerationResult, ToolError> {
        // This is a stub implementation.
        // In a real implementation, you would:
        // 1. Call an external image API (e.g., DALL-E, Stable Diffusion, Midjourney)
        // 2. Handle the response with image URL or base64 data
        // 3. Return structured result

        Ok(ImageGenerationResult {
            prompt: prompt.to_string(),
            image_url: Some("https://example.com/generated-image.png".to_string()),
            image_base64: None,
            revised_prompt: Some(format!("Enhanced version of: {}", prompt)),
            format: self.output_format.unwrap_or(OutputFormat::Png),
            width: 1024,
            height: 1024,
        })
    }
}

// ============================================================================
// Result Type
// ============================================================================

/// Result from an image generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResult {
    /// The original prompt.
    pub prompt: String,
    /// URL to the generated image (if hosted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    /// Base64-encoded image data (if inline).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<String>,
    /// Revised/enhanced prompt used by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    /// Output format.
    pub format: OutputFormat,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

// ============================================================================
// Tool Implementation
// ============================================================================

#[async_trait]
impl<Deps: Send + Sync> Tool<Deps> for ImageGenerationTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("image_generation", "Generate images from text prompts using AI")
            .with_parameters(Self::schema())
    }

    async fn call(&self, _ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_args("Missing 'prompt' field"))?;

        if prompt.trim().is_empty() {
            return Err(ToolError::invalid_args("Prompt cannot be empty"));
        }

        // Apply any overrides from args
        let mut tool = self.clone();

        if let Some(bg) = args.get("background").and_then(|v| v.as_str()) {
            tool.background = match bg {
                "transparent" => ImageBackground::Transparent,
                "opaque" => ImageBackground::Opaque,
                _ => ImageBackground::Auto,
            };
        }

        if let Some(fmt) = args.get("output_format").and_then(|v| v.as_str()) {
            tool.output_format = Some(match fmt {
                "webp" => OutputFormat::Webp,
                "jpeg" => OutputFormat::Jpeg,
                _ => OutputFormat::Png,
            });
        }

        if let Some(q) = args.get("quality").and_then(|v| v.as_str()) {
            tool.quality = match q {
                "low" => ImageQuality::Low,
                "medium" => ImageQuality::Medium,
                "high" => ImageQuality::High,
                _ => ImageQuality::Auto,
            };
        }

        if let Some(ratio) = args.get("aspect_ratio").and_then(|v| v.as_str()) {
            tool.aspect_ratio = Some(match ratio {
                "21_9" => ImageAspectRatio::R21_9,
                "16_9" => ImageAspectRatio::R16_9,
                "3_2" => ImageAspectRatio::R3_2,
                "4_3" => ImageAspectRatio::R4_3,
                "3_4" => ImageAspectRatio::R3_4,
                "2_3" => ImageAspectRatio::R2_3,
                "9_16" => ImageAspectRatio::R9_16,
                "9_21" => ImageAspectRatio::R9_21,
                _ => ImageAspectRatio::R1_1,
            });
        }

        if let Some(size) = args.get("size").and_then(|v| v.as_str()) {
            tool.size = Some(match size {
                "256x256" => ImageSize::Size256x256,
                "512x512" => ImageSize::Size512x512,
                "1024x1024" => ImageSize::Size1024x1024,
                "1024x1792" => ImageSize::Size1024x1792,
                "1792x1024" => ImageSize::Size1792x1024,
                "2048x2048" => ImageSize::Size2048x2048,
                _ => ImageSize::Auto,
            });
        }

        if let Some(comp) = args.get("output_compression").and_then(|v| v.as_u64()) {
            tool.output_compression = Some((comp as u8).min(100));
        }

        let result = tool.generate(prompt).await?;

        let output = serde_json::json!({
            "success": true,
            "result": result
        });

        Ok(ToolReturn::json(output))
    }

    fn max_retries(&self) -> Option<u32> {
        Some(2)
    }
}

// ============================================================================
// Provider Trait
// ============================================================================

/// Trait for image generation providers.
#[allow(async_fn_in_trait)]
pub trait ImageGenerationProvider: Send + Sync {
    /// Generate an image from a prompt.
    async fn generate(
        &self,
        prompt: &str,
        tool: &ImageGenerationTool,
    ) -> Result<ImageGenerationResult, ToolError>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_background_default() {
        assert_eq!(ImageBackground::default(), ImageBackground::Auto);
    }

    #[test]
    fn test_image_background_display() {
        assert_eq!(ImageBackground::Transparent.to_string(), "transparent");
        assert_eq!(ImageBackground::Opaque.to_string(), "opaque");
        assert_eq!(ImageBackground::Auto.to_string(), "auto");
    }

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Png);
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Png.to_string(), "png");
        assert_eq!(OutputFormat::Webp.to_string(), "webp");
        assert_eq!(OutputFormat::Jpeg.to_string(), "jpeg");
    }

    #[test]
    fn test_image_quality_default() {
        assert_eq!(ImageQuality::default(), ImageQuality::Auto);
    }

    #[test]
    fn test_image_quality_display() {
        assert_eq!(ImageQuality::Low.to_string(), "low");
        assert_eq!(ImageQuality::Medium.to_string(), "medium");
        assert_eq!(ImageQuality::High.to_string(), "high");
        assert_eq!(ImageQuality::Auto.to_string(), "auto");
    }

    #[test]
    fn test_image_aspect_ratio_default() {
        assert_eq!(ImageAspectRatio::default(), ImageAspectRatio::R1_1);
    }

    #[test]
    fn test_image_aspect_ratio_display() {
        assert_eq!(ImageAspectRatio::R21_9.to_string(), "21:9");
        assert_eq!(ImageAspectRatio::R16_9.to_string(), "16:9");
        assert_eq!(ImageAspectRatio::R1_1.to_string(), "1:1");
        assert_eq!(ImageAspectRatio::R9_16.to_string(), "9:16");
    }

    #[test]
    fn test_image_size_default() {
        assert_eq!(ImageSize::default(), ImageSize::Auto);
    }

    #[test]
    fn test_image_size_display() {
        assert_eq!(ImageSize::Auto.to_string(), "auto");
        assert_eq!(ImageSize::Size1024x1024.to_string(), "1024x1024");
        assert_eq!(ImageSize::Size1792x1024.to_string(), "1792x1024");
    }

    #[test]
    fn test_image_generation_tool_default() {
        let tool = ImageGenerationTool::new();
        assert_eq!(tool.background, ImageBackground::Auto);
        assert_eq!(tool.quality, ImageQuality::Auto);
        assert!(tool.output_format.is_none());
        assert!(tool.aspect_ratio.is_none());
        assert!(tool.size.is_none());
        assert!(tool.output_compression.is_none());
        assert_eq!(tool.partial_images, 0);
    }

    #[test]
    fn test_image_generation_tool_builder() {
        let tool = ImageGenerationTool::new()
            .background(ImageBackground::Transparent)
            .output_format(OutputFormat::Webp)
            .quality(ImageQuality::High)
            .aspect_ratio(ImageAspectRatio::R16_9)
            .size(ImageSize::Size1024x1024)
            .output_compression(80)
            .partial_images(2);

        assert_eq!(tool.background, ImageBackground::Transparent);
        assert_eq!(tool.output_format, Some(OutputFormat::Webp));
        assert_eq!(tool.quality, ImageQuality::High);
        assert_eq!(tool.aspect_ratio, Some(ImageAspectRatio::R16_9));
        assert_eq!(tool.size, Some(ImageSize::Size1024x1024));
        assert_eq!(tool.output_compression, Some(80));
        assert_eq!(tool.partial_images, 2);
    }

    #[test]
    fn test_output_compression_clamping() {
        let tool = ImageGenerationTool::new().output_compression(150);
        assert_eq!(tool.output_compression, Some(100));
    }

    #[test]
    fn test_partial_images_clamping() {
        let tool = ImageGenerationTool::new().partial_images(10);
        assert_eq!(tool.partial_images, 3);
    }

    #[test]
    fn test_image_generation_tool_definition() {
        let tool = ImageGenerationTool::new();
        let def = <ImageGenerationTool as Tool<()>>::definition(&tool);
        assert_eq!(def.name, "image_generation");
        assert!(def.parameters().is_required("prompt"));
    }

    #[tokio::test]
    async fn test_image_generation_tool_call() {
        let tool = ImageGenerationTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(&ctx, serde_json::json!({"prompt": "a cute puppy"}))
            .await
            .unwrap();

        assert!(!result.is_error());
        let json = result.as_json().unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["result"]["prompt"], "a cute puppy");
    }

    #[tokio::test]
    async fn test_image_generation_missing_prompt() {
        let tool = ImageGenerationTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool.call(&ctx, serde_json::json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn test_image_generation_empty_prompt() {
        let tool = ImageGenerationTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool.call(&ctx, serde_json::json!({"prompt": "  "})).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn test_image_generation_with_options() {
        let tool = ImageGenerationTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(
                &ctx,
                serde_json::json!({
                    "prompt": "a sunset over mountains",
                    "background": "transparent",
                    "output_format": "webp",
                    "quality": "high",
                    "aspect_ratio": "16_9",
                    "size": "1024x1024"
                }),
            )
            .await
            .unwrap();

        assert!(!result.is_error());
    }

    #[test]
    fn test_serde_roundtrip() {
        let tool = ImageGenerationTool::new()
            .background(ImageBackground::Transparent)
            .quality(ImageQuality::High)
            .output_format(OutputFormat::Png);

        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ImageGenerationTool = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.background, ImageBackground::Transparent);
        assert_eq!(deserialized.quality, ImageQuality::High);
        assert_eq!(deserialized.output_format, Some(OutputFormat::Png));
    }

    #[test]
    fn test_image_generation_result_serde() {
        let result = ImageGenerationResult {
            prompt: "test prompt".to_string(),
            image_url: Some("https://example.com/image.png".to_string()),
            image_base64: None,
            revised_prompt: Some("enhanced prompt".to_string()),
            format: OutputFormat::Png,
            width: 1024,
            height: 1024,
        };

        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["prompt"], "test prompt");
        assert_eq!(json["width"], 1024);
        assert!(json.get("image_base64").is_none()); // skipped because None
    }
}
