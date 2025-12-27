//! Multi-modal content types for user prompts.
//!
//! This module defines the content types that can be included in user messages,
//! supporting text, images, audio, video, documents, and generic files.

use serde::{Deserialize, Serialize};

use super::media::{AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType};

/// User message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// Plain text content.
    Text(String),
    /// Multi-part content.
    Parts(Vec<UserContentPart>),
}

impl UserContent {
    /// Create text content.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create multi-part content.
    #[must_use]
    pub fn parts(parts: Vec<UserContentPart>) -> Self {
        Self::Parts(parts)
    }

    /// Check if this is text content.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Get as text if this is text content.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get all parts (wraps text in a single-element vec if needed).
    #[must_use]
    pub fn to_parts(&self) -> Vec<UserContentPart> {
        match self {
            Self::Text(s) => vec![UserContentPart::Text { text: s.clone() }],
            Self::Parts(parts) => parts.clone(),
        }
    }
}

impl Default for UserContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for UserContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<Vec<UserContentPart>> for UserContent {
    fn from(parts: Vec<UserContentPart>) -> Self {
        Self::Parts(parts)
    }
}

/// Individual content part in a multi-part message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContentPart {
    /// Text content.
    Text {
        /// The text.
        text: String,
    },
    /// Image content.
    Image {
        /// The image.
        #[serde(flatten)]
        image: ImageContent,
    },
    /// Audio content.
    Audio {
        /// The audio.
        #[serde(flatten)]
        audio: AudioContent,
    },
    /// Video content.
    Video {
        /// The video.
        #[serde(flatten)]
        video: VideoContent,
    },
    /// Document content.
    Document {
        /// The document.
        #[serde(flatten)]
        document: DocumentContent,
    },
    /// Generic file content.
    File {
        /// The file.
        #[serde(flatten)]
        file: FileContent,
    },
}

impl UserContentPart {
    /// Create text content.
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Create image content from URL.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image { image: ImageContent::url(url) }
    }

    /// Create image content from binary data.
    #[must_use]
    pub fn image_binary(data: Vec<u8>, media_type: ImageMediaType) -> Self {
        Self::Image { image: ImageContent::binary(data, media_type) }
    }
}

/// Image content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImageContent {
    /// Image from URL.
    Url(ImageUrl),
    /// Binary image data.
    Binary(BinaryImage),
}

impl ImageContent {
    /// Create from URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(ImageUrl::new(url))
    }

    /// Create from binary data.
    #[must_use]
    pub fn binary(data: Vec<u8>, media_type: ImageMediaType) -> Self {
        Self::Binary(BinaryImage::new(data, media_type))
    }

    /// Get the media type if known.
    #[must_use]
    pub fn media_type(&self) -> Option<ImageMediaType> {
        match self {
            Self::Url(u) => u.media_type,
            Self::Binary(b) => Some(b.media_type),
        }
    }
}

/// Image from URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The image URL.
    pub url: String,
    /// Media type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<ImageMediaType>,
    /// Force download instead of using URL directly.
    #[serde(default)]
    pub force_download: bool,
    /// Vendor-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_metadata: Option<serde_json::Value>,
}

impl ImageUrl {
    /// Create a new image URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            media_type: None,
            force_download: false,
            vendor_metadata: None,
        }
    }

    /// Set the media type.
    #[must_use]
    pub fn with_media_type(mut self, media_type: ImageMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }

    /// Set force download.
    #[must_use]
    pub fn with_force_download(mut self, force: bool) -> Self {
        self.force_download = force;
        self
    }

    /// Set vendor metadata.
    #[must_use]
    pub fn with_vendor_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.vendor_metadata = Some(metadata);
        self
    }
}

/// Binary image data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryImage {
    /// The raw image data.
    #[serde(with = "base64_serde")]
    pub data: Vec<u8>,
    /// The media type.
    pub media_type: ImageMediaType,
}

impl BinaryImage {
    /// Create new binary image.
    #[must_use]
    pub fn new(data: Vec<u8>, media_type: ImageMediaType) -> Self {
        Self { data, media_type }
    }

    /// Get data as base64 string.
    #[must_use]
    pub fn to_base64(&self) -> String {
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &self.data)
    }

    /// Get as data URL.
    #[must_use]
    pub fn to_data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type.mime_type(), self.to_base64())
    }
}

/// Audio content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AudioContent {
    /// Audio from URL.
    Url(AudioUrl),
    /// Binary audio data.
    Binary(BinaryAudio),
}

impl AudioContent {
    /// Create from URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(AudioUrl::new(url))
    }

    /// Create from binary data.
    #[must_use]
    pub fn binary(data: Vec<u8>, media_type: AudioMediaType) -> Self {
        Self::Binary(BinaryAudio::new(data, media_type))
    }
}

/// Audio from URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioUrl {
    /// The audio URL.
    pub url: String,
    /// Media type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<AudioMediaType>,
    /// Force download instead of using URL directly.
    #[serde(default)]
    pub force_download: bool,
    /// Vendor-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_metadata: Option<serde_json::Value>,
}

impl AudioUrl {
    /// Create a new audio URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            media_type: None,
            force_download: false,
            vendor_metadata: None,
        }
    }

    /// Set the media type.
    #[must_use]
    pub fn with_media_type(mut self, media_type: AudioMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }
}

/// Binary audio data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryAudio {
    /// The raw audio data.
    #[serde(with = "base64_serde")]
    pub data: Vec<u8>,
    /// The media type.
    pub media_type: AudioMediaType,
}

impl BinaryAudio {
    /// Create new binary audio.
    #[must_use]
    pub fn new(data: Vec<u8>, media_type: AudioMediaType) -> Self {
        Self { data, media_type }
    }
}

/// Video content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VideoContent {
    /// Video from URL.
    Url(VideoUrl),
    /// Binary video data.
    Binary(BinaryVideo),
}

impl VideoContent {
    /// Create from URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(VideoUrl::new(url))
    }

    /// Create from binary data.
    #[must_use]
    pub fn binary(data: Vec<u8>, media_type: VideoMediaType) -> Self {
        Self::Binary(BinaryVideo::new(data, media_type))
    }
}

/// Video from URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoUrl {
    /// The video URL.
    pub url: String,
    /// Media type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<VideoMediaType>,
    /// Force download instead of using URL directly.
    #[serde(default)]
    pub force_download: bool,
    /// Vendor-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_metadata: Option<serde_json::Value>,
}

impl VideoUrl {
    /// Create a new video URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            media_type: None,
            force_download: false,
            vendor_metadata: None,
        }
    }

    /// Set the media type.
    #[must_use]
    pub fn with_media_type(mut self, media_type: VideoMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }
}

/// Binary video data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryVideo {
    /// The raw video data.
    #[serde(with = "base64_serde")]
    pub data: Vec<u8>,
    /// The media type.
    pub media_type: VideoMediaType,
}

impl BinaryVideo {
    /// Create new binary video.
    #[must_use]
    pub fn new(data: Vec<u8>, media_type: VideoMediaType) -> Self {
        Self { data, media_type }
    }
}

/// Document content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentContent {
    /// Document from URL.
    Url(DocumentUrl),
    /// Binary document data.
    Binary(BinaryDocument),
}

impl DocumentContent {
    /// Create from URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(DocumentUrl::new(url))
    }

    /// Create from binary data.
    #[must_use]
    pub fn binary(data: Vec<u8>, media_type: DocumentMediaType) -> Self {
        Self::Binary(BinaryDocument::new(data, media_type))
    }
}

/// Document from URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentUrl {
    /// The document URL.
    pub url: String,
    /// Media type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<DocumentMediaType>,
    /// Force download instead of using URL directly.
    #[serde(default)]
    pub force_download: bool,
    /// Vendor-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_metadata: Option<serde_json::Value>,
}

impl DocumentUrl {
    /// Create a new document URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            media_type: None,
            force_download: false,
            vendor_metadata: None,
        }
    }

    /// Set the media type.
    #[must_use]
    pub fn with_media_type(mut self, media_type: DocumentMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }
}

/// Binary document data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryDocument {
    /// The raw document data.
    #[serde(with = "base64_serde")]
    pub data: Vec<u8>,
    /// The media type.
    pub media_type: DocumentMediaType,
    /// Optional filename.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

impl BinaryDocument {
    /// Create new binary document.
    #[must_use]
    pub fn new(data: Vec<u8>, media_type: DocumentMediaType) -> Self {
        Self {
            data,
            media_type,
            filename: None,
        }
    }

    /// Set the filename.
    #[must_use]
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }
}

/// Generic file content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileContent {
    /// File from URL.
    Url(FileUrl),
    /// Binary file data.
    Binary(BinaryFile),
}

impl FileContent {
    /// Create from URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(FileUrl::new(url))
    }

    /// Create from binary data.
    #[must_use]
    pub fn binary(data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self::Binary(BinaryFile::new(data, mime_type))
    }
}

/// File from URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileUrl {
    /// The file URL.
    pub url: String,
    /// MIME type hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Force download instead of using URL directly.
    #[serde(default)]
    pub force_download: bool,
    /// Vendor-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_metadata: Option<serde_json::Value>,
}

impl FileUrl {
    /// Create a new file URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            mime_type: None,
            force_download: false,
            vendor_metadata: None,
        }
    }

    /// Set the MIME type.
    #[must_use]
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

/// Binary file data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryFile {
    /// The raw file data.
    #[serde(with = "base64_serde")]
    pub data: Vec<u8>,
    /// The MIME type.
    pub mime_type: String,
    /// Optional filename.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

impl BinaryFile {
    /// Create new binary file.
    #[must_use]
    pub fn new(data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            data,
            mime_type: mime_type.into(),
            filename: None,
        }
    }

    /// Set the filename.
    #[must_use]
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }
}

/// Serde helper for base64 encoding.
mod base64_serde {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(data))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        STANDARD.decode(s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_content_text() {
        let content = UserContent::text("Hello, world!");
        assert!(content.is_text());
        assert_eq!(content.as_text(), Some("Hello, world!"));
    }

    #[test]
    fn test_user_content_from_string() {
        let content: UserContent = "Hello".into();
        assert!(content.is_text());
    }

    #[test]
    fn test_image_url() {
        let img = ImageUrl::new("https://example.com/image.png")
            .with_media_type(ImageMediaType::Png);
        assert_eq!(img.url, "https://example.com/image.png");
        assert_eq!(img.media_type, Some(ImageMediaType::Png));
    }

    #[test]
    fn test_binary_image_to_data_url() {
        let img = BinaryImage::new(vec![1, 2, 3, 4], ImageMediaType::Png);
        let data_url = img.to_data_url();
        assert!(data_url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let content = UserContent::parts(vec![
            UserContentPart::text("Hello"),
            UserContentPart::image_url("https://example.com/img.jpg"),
        ]);
        let json = serde_json::to_string(&content).unwrap();
        let parsed: UserContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, parsed);
    }
}
