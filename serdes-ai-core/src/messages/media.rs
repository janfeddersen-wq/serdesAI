//! Media type enums for multi-modal content.
//!
//! This module defines all supported media types for images, audio, video,
//! and documents, along with MIME type conversion utilities.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Image media types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageMediaType {
    /// JPEG image.
    Jpeg,
    /// PNG image.
    #[default]
    Png,
    /// GIF image.
    Gif,
    /// WebP image.
    Webp,
}

impl ImageMediaType {
    /// Get the MIME type string.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }

    /// Get the file extension.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Gif => "gif",
            Self::Webp => "webp",
        }
    }

    /// Try to detect from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "png" => Some(Self::Png),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }
}

impl fmt::Display for ImageMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mime_type())
    }
}

impl FromStr for ImageMediaType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "image/jpeg" | "jpeg" | "jpg" => Ok(Self::Jpeg),
            "image/png" | "png" => Ok(Self::Png),
            "image/gif" | "gif" => Ok(Self::Gif),
            "image/webp" | "webp" => Ok(Self::Webp),
            _ => Err(format!("Unknown image media type: {}", s)),
        }
    }
}

/// Audio media types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioMediaType {
    /// WAV audio.
    Wav,
    /// MPEG audio (MP3).
    #[default]
    Mpeg,
    /// Ogg Vorbis audio.
    Ogg,
    /// FLAC audio.
    Flac,
    /// AIFF audio.
    Aiff,
    /// AAC audio.
    Aac,
    /// WebM audio.
    Webm,
}

impl AudioMediaType {
    /// Get the MIME type string.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mpeg => "audio/mpeg",
            Self::Ogg => "audio/ogg",
            Self::Flac => "audio/flac",
            Self::Aiff => "audio/aiff",
            Self::Aac => "audio/aac",
            Self::Webm => "audio/webm",
        }
    }

    /// Get the file extension.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mpeg => "mp3",
            Self::Ogg => "ogg",
            Self::Flac => "flac",
            Self::Aiff => "aiff",
            Self::Aac => "aac",
            Self::Webm => "webm",
        }
    }

    /// Try to detect from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "mp3" => Some(Self::Mpeg),
            "ogg" => Some(Self::Ogg),
            "flac" => Some(Self::Flac),
            "aiff" | "aif" => Some(Self::Aiff),
            "aac" | "m4a" => Some(Self::Aac),
            "webm" => Some(Self::Webm),
            _ => None,
        }
    }
}

impl fmt::Display for AudioMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mime_type())
    }
}

impl FromStr for AudioMediaType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "audio/wav" | "audio/wave" | "wav" => Ok(Self::Wav),
            "audio/mpeg" | "audio/mp3" | "mp3" | "mpeg" => Ok(Self::Mpeg),
            "audio/ogg" | "ogg" => Ok(Self::Ogg),
            "audio/flac" | "flac" => Ok(Self::Flac),
            "audio/aiff" | "aiff" => Ok(Self::Aiff),
            "audio/aac" | "aac" => Ok(Self::Aac),
            "audio/webm" | "webm" => Ok(Self::Webm),
            _ => Err(format!("Unknown audio media type: {}", s)),
        }
    }
}

/// Video media types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoMediaType {
    /// MP4 video.
    #[default]
    Mp4,
    /// WebM video.
    Webm,
    /// QuickTime video.
    Mov,
    /// Matroska video.
    Mkv,
    /// Flash video.
    Flv,
    /// MPEG video.
    Mpeg,
    /// Windows Media Video.
    Wmv,
    /// 3GPP video.
    ThreeGp,
    /// AVI video.
    Avi,
}

impl VideoMediaType {
    /// Get the MIME type string.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::Webm => "video/webm",
            Self::Mov => "video/quicktime",
            Self::Mkv => "video/x-matroska",
            Self::Flv => "video/x-flv",
            Self::Mpeg => "video/mpeg",
            Self::Wmv => "video/x-ms-wmv",
            Self::ThreeGp => "video/3gpp",
            Self::Avi => "video/x-msvideo",
        }
    }

    /// Get the file extension.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Webm => "webm",
            Self::Mov => "mov",
            Self::Mkv => "mkv",
            Self::Flv => "flv",
            Self::Mpeg => "mpeg",
            Self::Wmv => "wmv",
            Self::ThreeGp => "3gp",
            Self::Avi => "avi",
        }
    }

    /// Try to detect from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp4" | "m4v" => Some(Self::Mp4),
            "webm" => Some(Self::Webm),
            "mov" | "qt" => Some(Self::Mov),
            "mkv" => Some(Self::Mkv),
            "flv" => Some(Self::Flv),
            "mpeg" | "mpg" => Some(Self::Mpeg),
            "wmv" => Some(Self::Wmv),
            "3gp" | "3gpp" => Some(Self::ThreeGp),
            "avi" => Some(Self::Avi),
            _ => None,
        }
    }
}

impl fmt::Display for VideoMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mime_type())
    }
}

impl FromStr for VideoMediaType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "video/mp4" | "mp4" => Ok(Self::Mp4),
            "video/webm" | "webm" => Ok(Self::Webm),
            "video/quicktime" | "mov" => Ok(Self::Mov),
            "video/x-matroska" | "mkv" => Ok(Self::Mkv),
            "video/x-flv" | "flv" => Ok(Self::Flv),
            "video/mpeg" | "mpeg" => Ok(Self::Mpeg),
            "video/x-ms-wmv" | "wmv" => Ok(Self::Wmv),
            "video/3gpp" | "3gp" => Ok(Self::ThreeGp),
            "video/x-msvideo" | "avi" => Ok(Self::Avi),
            _ => Err(format!("Unknown video media type: {}", s)),
        }
    }
}

/// Document media types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentMediaType {
    /// PDF document.
    Pdf,
    /// Plain text.
    Plain,
    /// CSV file.
    Csv,
    /// Microsoft Word (DOCX).
    Docx,
    /// Microsoft Word (DOC).
    Doc,
    /// Microsoft Excel (XLSX).
    Xlsx,
    /// Microsoft Excel (XLS).
    Xls,
    /// HTML document.
    Html,
    /// Markdown document.
    Markdown,
    /// RTF document.
    Rtf,
    /// JSON document.
    Json,
    /// XML document.
    Xml,
}

impl DocumentMediaType {
    /// Get the MIME type string.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Pdf => "application/pdf",
            Self::Plain => "text/plain",
            Self::Csv => "text/csv",
            Self::Docx => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Self::Doc => "application/msword",
            Self::Xlsx => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Self::Xls => "application/vnd.ms-excel",
            Self::Html => "text/html",
            Self::Markdown => "text/markdown",
            Self::Rtf => "application/rtf",
            Self::Json => "application/json",
            Self::Xml => "application/xml",
        }
    }

    /// Get the file extension.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Plain => "txt",
            Self::Csv => "csv",
            Self::Docx => "docx",
            Self::Doc => "doc",
            Self::Xlsx => "xlsx",
            Self::Xls => "xls",
            Self::Html => "html",
            Self::Markdown => "md",
            Self::Rtf => "rtf",
            Self::Json => "json",
            Self::Xml => "xml",
        }
    }

    /// Try to detect from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "pdf" => Some(Self::Pdf),
            "txt" | "text" => Some(Self::Plain),
            "csv" => Some(Self::Csv),
            "docx" => Some(Self::Docx),
            "doc" => Some(Self::Doc),
            "xlsx" => Some(Self::Xlsx),
            "xls" => Some(Self::Xls),
            "html" | "htm" => Some(Self::Html),
            "md" | "markdown" => Some(Self::Markdown),
            "rtf" => Some(Self::Rtf),
            "json" => Some(Self::Json),
            "xml" => Some(Self::Xml),
            _ => None,
        }
    }

    /// Check if this is a text-based format.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(
            self,
            Self::Plain | Self::Csv | Self::Html | Self::Markdown | Self::Json | Self::Xml
        )
    }
}

impl fmt::Display for DocumentMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mime_type())
    }
}

impl FromStr for DocumentMediaType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "application/pdf" | "pdf" => Ok(Self::Pdf),
            "text/plain" | "plain" | "txt" => Ok(Self::Plain),
            "text/csv" | "csv" => Ok(Self::Csv),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" | "docx" => {
                Ok(Self::Docx)
            }
            "application/msword" | "doc" => Ok(Self::Doc),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" | "xlsx" => {
                Ok(Self::Xlsx)
            }
            "application/vnd.ms-excel" | "xls" => Ok(Self::Xls),
            "text/html" | "html" => Ok(Self::Html),
            "text/markdown" | "markdown" | "md" => Ok(Self::Markdown),
            "application/rtf" | "rtf" => Ok(Self::Rtf),
            "application/json" | "json" => Ok(Self::Json),
            "application/xml" | "text/xml" | "xml" => Ok(Self::Xml),
            _ => Err(format!("Unknown document media type: {}", s)),
        }
    }
}

impl Default for DocumentMediaType {
    fn default() -> Self {
        Self::Plain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_media_type() {
        assert_eq!(ImageMediaType::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(
            ImageMediaType::from_extension("png"),
            Some(ImageMediaType::Png)
        );
        assert_eq!(
            "image/webp".parse::<ImageMediaType>().unwrap(),
            ImageMediaType::Webp
        );
    }

    #[test]
    fn test_audio_media_type() {
        assert_eq!(AudioMediaType::Mpeg.extension(), "mp3");
        assert_eq!(
            AudioMediaType::from_extension("flac"),
            Some(AudioMediaType::Flac)
        );
    }

    #[test]
    fn test_document_is_text() {
        assert!(DocumentMediaType::Plain.is_text());
        assert!(DocumentMediaType::Json.is_text());
        assert!(!DocumentMediaType::Pdf.is_text());
        assert!(!DocumentMediaType::Docx.is_text());
    }

    #[test]
    fn test_serde_roundtrip() {
        let img = ImageMediaType::Png;
        let json = serde_json::to_string(&img).unwrap();
        assert_eq!(json, "\"png\"");
        let parsed: ImageMediaType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, img);
    }
}
