//! File extension analysis stage.
//!
//! This stage classifies based on file extensions and their size distribution.

use std::path::Path;

use crate::cascade::{Medium, Stage, StageResult};
use crate::{ContentInfo, Error};

/// Stage that analyzes file extensions to determine content type.
///
/// Returns `High` confidence when a single medium dominates by file size,
/// and the files have clear, unambiguous extensions.
pub struct ExtensionStage {
    /// Minimum percentage of total size for a medium to be considered dominant.
    dominance_threshold: f64,
}

impl ExtensionStage {
    /// Create a new extension stage with default settings.
    pub fn new() -> Self {
        Self {
            dominance_threshold: 0.9,
        }
    }

    /// Set the dominance threshold (0.0 to 1.0).
    ///
    /// A medium must account for this fraction of total file size to be
    /// considered a confident match.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.dominance_threshold = threshold.clamp(0.5, 1.0);
        self
    }

    /// Analyze files and return size by medium category.
    fn analyze_files(&self, info: &ContentInfo) -> SizeAnalysis {
        let mut analysis = SizeAnalysis::default();

        for file in &info.files {
            let ext = get_extension(&file.filename);
            let category = categorize_extension(ext);

            match category {
                ExtCategory::Video => analysis.video += file.size,
                ExtCategory::Audio => analysis.audio += file.size,
                ExtCategory::Book => analysis.book += file.size,
                ExtCategory::Software => analysis.software += file.size,
                ExtCategory::Archive => analysis.archive += file.size,
                ExtCategory::Other => analysis.other += file.size,
            }

            analysis.total += file.size;
        }

        analysis
    }
}

impl Default for ExtensionStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for ExtensionStage {
    fn name(&self) -> &'static str {
        "extensions"
    }

    fn classify(&self, info: &ContentInfo) -> Result<Option<StageResult>, Error> {
        // Need files to analyze
        if info.files.is_empty() {
            return Ok(None);
        }

        let analysis = self.analyze_files(info);

        // Need non-zero total size
        if analysis.total == 0 {
            return Ok(None);
        }

        // Calculate known content (excluding archives and other)
        let known_content = analysis.video + analysis.audio + analysis.book + analysis.software;

        // If we have known content, calculate dominance excluding archives/other
        // This handles cases like "book.pdf + book.djvu + metadata.zip"
        let (effective_total, use_known) = if known_content > 0 {
            (known_content as f64, true)
        } else {
            (analysis.total as f64, false)
        };

        let threshold = self.dominance_threshold;

        // Video dominance
        if analysis.video > 0 {
            let ratio = analysis.video as f64 / effective_total;
            if ratio >= threshold {
                return Ok(Some(StageResult::high(Medium::Video, "extensions")));
            }
        }

        // Audio dominance
        if analysis.audio > 0 {
            let ratio = analysis.audio as f64 / effective_total;
            if ratio >= threshold {
                return Ok(Some(StageResult::high(Medium::Audio, "extensions")));
            }
        }

        // Book dominance
        if analysis.book > 0 {
            let ratio = analysis.book as f64 / effective_total;
            if ratio >= threshold {
                return Ok(Some(StageResult::high(Medium::Book, "extensions")));
            }
        }

        // Software dominance
        if analysis.software > 0 {
            let ratio = analysis.software as f64 / effective_total;
            if ratio >= threshold {
                return Ok(Some(StageResult::high(Medium::Software, "extensions")));
            }
        }

        // If we only have one type of known content but it didn't meet threshold,
        // still return it with medium confidence
        if use_known {
            let categories = [
                (analysis.video, Medium::Video),
                (analysis.audio, Medium::Audio),
                (analysis.book, Medium::Book),
                (analysis.software, Medium::Software),
            ];

            let non_zero: Vec<_> = categories.iter().filter(|(size, _)| *size > 0).collect();
            if non_zero.len() == 1 {
                return Ok(Some(StageResult::medium(non_zero[0].1.clone(), "extensions")));
            }
        }

        // No clear dominance - pass to next stage
        Ok(None)
    }
}

/// Size distribution by category.
#[derive(Debug, Default)]
struct SizeAnalysis {
    video: u64,
    audio: u64,
    book: u64,
    software: u64,
    archive: u64,
    other: u64,
    total: u64,
}

/// Extension category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtCategory {
    Video,
    Audio,
    Book,
    Software,
    Archive,
    Other,
}

/// Get lowercase extension from filename.
fn get_extension(filename: &str) -> &str {
    Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
}

/// Categorize a file extension.
fn categorize_extension(ext: &str) -> ExtCategory {
    let ext = ext.to_lowercase();

    match ext.as_str() {
        // Video
        "mkv" | "mp4" | "avi" | "wmv" | "mov" | "m4v" | "ts" | "m2ts" | "webm" | "flv" | "vob"
        | "ogv" | "3gp" => ExtCategory::Video,

        // Audio
        "mp3" | "flac" | "ogg" | "opus" | "wav" | "aac" | "m4a" | "wma" | "ape" | "alac"
        | "aiff" | "mid" | "midi" => ExtCategory::Audio,

        // Book
        "epub" | "mobi" | "azw" | "azw3" | "djvu" | "cbr" | "cbz" | "cb7" | "pdf" => {
            ExtCategory::Book
        }

        // Software
        "exe" | "msi" | "dmg" | "pkg" | "deb" | "rpm" | "apk" | "ipa" | "app" | "dll" | "so"
        | "dylib" => ExtCategory::Software,

        // Archives (ambiguous - could contain anything)
        "zip" | "rar" | "7z" | "tar" | "gz" | "xz" | "bz2" | "iso" | "img" => ExtCategory::Archive,

        // Everything else
        _ => ExtCategory::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::Confidence;
    use crate::FileInfo;

    fn make_file(name: &str, size: u64) -> FileInfo {
        FileInfo {
            path: name.to_string(),
            filename: name.to_string(),
            size,
        }
    }

    #[test]
    fn test_video_dominance() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "Movie.2024".to_string(),
            files: vec![
                make_file("movie.mkv", 1_000_000_000), // 1GB video
                make_file("movie.srt", 50_000),        // 50KB subtitle
            ],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.medium, Medium::Video);
        assert_eq!(result.confidence, Confidence::High);
    }

    #[test]
    fn test_audio_dominance() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "Artist - Album".to_string(),
            files: vec![
                make_file("01-track.mp3", 10_000_000),
                make_file("02-track.mp3", 10_000_000),
                make_file("cover.jpg", 100_000),
            ],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.medium, Medium::Audio);
        assert_eq!(result.confidence, Confidence::High);
    }

    #[test]
    fn test_book_dominance() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "Book Collection".to_string(),
            files: vec![
                make_file("book1.epub", 1_000_000),
                make_file("book2.epub", 1_000_000),
                make_file("readme.txt", 1_000),
            ],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.medium, Medium::Book);
    }

    #[test]
    fn test_software_dominance() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "App Setup".to_string(),
            files: vec![
                make_file("setup.exe", 500_000_000),
                make_file("readme.txt", 5_000),
            ],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.medium, Medium::Software);
    }

    #[test]
    fn test_mixed_content_no_match() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "Mixed Content".to_string(),
            files: vec![
                make_file("video.mkv", 500_000_000),
                make_file("audio.mp3", 400_000_000),
            ],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_none()); // No clear dominance
    }

    #[test]
    fn test_archive_not_classified() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "Archive".to_string(),
            files: vec![make_file("data.iso", 4_000_000_000)],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_none()); // ISO is ambiguous
    }

    #[test]
    fn test_empty_files_no_match() {
        let stage = ExtensionStage::new();
        let info = ContentInfo {
            name: "Empty".to_string(),
            files: vec![],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_extension_categorization() {
        assert_eq!(categorize_extension("mkv"), ExtCategory::Video);
        assert_eq!(categorize_extension("MP3"), ExtCategory::Audio);
        assert_eq!(categorize_extension("epub"), ExtCategory::Book);
        assert_eq!(categorize_extension("exe"), ExtCategory::Software);
        assert_eq!(categorize_extension("iso"), ExtCategory::Archive);
        assert_eq!(categorize_extension("xyz"), ExtCategory::Other);
    }
}
