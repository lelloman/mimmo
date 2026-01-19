//! Name pattern matching stage.
//!
//! This stage classifies based on patterns commonly found in torrent names,
//! combined with file extension hints for validation.

use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

use crate::cascade::{Medium, Stage, StageResult};
use crate::{ContentInfo, Error};

/// Stage that analyzes torrent names for classification patterns.
///
/// This stage looks for:
/// - Video: resolution (1080p, 4K), codecs (x264, HEVC), release tags
/// - Audio: bitrate (320kbps, FLAC), album patterns
/// - Book: common ebook naming patterns
/// - Software: version numbers, platform tags (x64, Win, Mac)
///
/// It also validates patterns against file extensions when available.
pub struct PatternStage {
    _private: (),
}

impl PatternStage {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for PatternStage {
    fn default() -> Self {
        Self::new()
    }
}

/// File type hints derived from extensions
#[derive(Debug, Default)]
struct FileHints {
    has_video: bool,
    has_audio: bool,
    has_book: bool,
    has_software: bool,
    has_archive: bool,
}

fn get_file_hints(info: &ContentInfo) -> FileHints {
    let mut hints = FileHints::default();

    for file in &info.files {
        let ext = Path::new(&file.filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "mkv" | "mp4" | "avi" | "wmv" | "mov" | "m4v" | "ts" | "m2ts" | "webm" | "flv"
            | "vob" | "ogv" | "3gp" => hints.has_video = true,
            "mp3" | "flac" | "ogg" | "opus" | "wav" | "aac" | "m4a" | "wma" | "ape" | "alac"
            | "aiff" => hints.has_audio = true,
            "epub" | "mobi" | "azw" | "azw3" | "djvu" | "cbr" | "cbz" | "cb7" | "pdf" => {
                hints.has_book = true
            }
            "exe" | "msi" | "dmg" | "pkg" | "deb" | "rpm" | "apk" | "ipa" | "iso" => {
                hints.has_software = true
            }
            "zip" | "rar" | "7z" | "tar" | "gz" => hints.has_archive = true,
            _ => {}
        }
    }

    hints
}

impl Stage for PatternStage {
    fn name(&self) -> &'static str {
        "patterns"
    }

    fn classify(&self, info: &ContentInfo) -> Result<Option<StageResult>, Error> {
        let name = &info.name;
        let hints = get_file_hints(info);

        // Check patterns in order of specificity, validating against file hints
        if let Some(result) = check_video_patterns(name, &hints) {
            return Ok(Some(result));
        }

        if let Some(result) = check_audio_patterns(name, &hints) {
            return Ok(Some(result));
        }

        if let Some(result) = check_software_patterns(name, &hints) {
            return Ok(Some(result));
        }

        if let Some(result) = check_book_patterns(name, &hints) {
            return Ok(Some(result));
        }

        Ok(None)
    }
}

// Video pattern checking
fn check_video_patterns(name: &str, hints: &FileHints) -> Option<StageResult> {
    static VIDEO_HIGH: OnceLock<Regex> = OnceLock::new();
    static VIDEO_MEDIUM: OnceLock<Regex> = OnceLock::new();

    // If we have file hints and they show audio but no video, skip video patterns
    // (e.g., "Lucy - Luc Besson (2014)" with .avi should match, but with .mp3 should not)
    if hints.has_audio && !hints.has_video {
        return None;
    }

    // High confidence: resolution + codec or release group patterns
    let video_high = VIDEO_HIGH.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Resolution patterns with codec/source
            (2160p|1080p|720p|480p|4K|UHD).*?(x264|x265|HEVC|H\.?264|H\.?265|AVC|BluRay|BDRip|WEB-?DL|WEB-?Rip|HDRip|DVDRip|HDTV)
            |
            # Codec/source patterns with resolution
            (x264|x265|HEVC|H\.?264|H\.?265|BluRay|BDRip|WEB-?DL|WEB-?Rip).*?(2160p|1080p|720p|480p|4K|UHD)
            |
            # Season/Episode patterns (TV shows)
            S\d{1,2}E\d{1,2}|Season\s*\d+|Complete\s+Series
            |
            # Year in brackets/parens with video indicators
            \(\d{4}\).*?(1080p|720p|BluRay|WEB-?DL|HDTV)
            |
            (1080p|720p|BluRay|WEB-?DL|HDTV).*?\(\d{4}\)
            "
        ).unwrap()
    });

    // Medium confidence: single strong video indicator
    // But require video files or no files at all (name-only classification)
    let video_medium = VIDEO_MEDIUM.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Single resolution marker
            \b(2160p|1080p|720p|480p|4K|UHD)\b
            |
            # Single codec marker with word boundary
            \b(x264|x265|HEVC)\b
            |
            # Single source marker
            \b(BluRay|BDRip|WEB-?DL|WEB-?Rip|HDRip|DVDRip|HDTV)\b
            |
            # Audio codec markers commonly in video releases
            \b(DTS-HD|Atmos|TrueHD|DD5\.?1)\b
            "
        ).unwrap()
    });

    if video_high.is_match(name) {
        return Some(StageResult::high(Medium::Video, "patterns"));
    }

    // For medium confidence, require video file hints if we have any file info
    if video_medium.is_match(name) {
        if hints.has_video || (!hints.has_audio && !hints.has_book && !hints.has_software) {
            return Some(StageResult::medium(Medium::Video, "patterns"));
        }
    }

    None
}

// Audio pattern checking
fn check_audio_patterns(name: &str, hints: &FileHints) -> Option<StageResult> {
    static AUDIO_HIGH: OnceLock<Regex> = OnceLock::new();
    static AUDIO_MEDIUM: OnceLock<Regex> = OnceLock::new();

    // If we have video files but no audio files, skip audio patterns
    if hints.has_video && !hints.has_audio {
        return None;
    }

    // High confidence: album/music specific patterns
    let audio_high = AUDIO_HIGH.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Album patterns with format
            (FLAC|MP3|320kbps|V0|V2|ALAC|24bit|16bit).*?(Album|Discography|OST|Soundtrack|EP\b)
            |
            (Album|Discography|OST|Soundtrack|EP\b).*?(FLAC|MP3|320kbps|V0|V2|ALAC)
            |
            # CD rip patterns
            \b(CD|CDDA|Lossless)\b.*?(FLAC|WAV|APE)
            |
            # Bitrate + format patterns
            \b(320|256|192|128)\s*kbps\b
            |
            # Music release group patterns
            \[FLAC\]|\[MP3\]|\[320\]
            "
        ).unwrap()
    });

    // Medium confidence: single audio indicator
    let audio_medium = AUDIO_MEDIUM.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Discography mention (strong signal)
            \bDiscography\b
            |
            # OST/Soundtrack (but validate with hints)
            \b(OST|Soundtrack)\b
            "
        ).unwrap()
    });

    if audio_high.is_match(name) {
        // High confidence patterns are strong enough on their own
        return Some(StageResult::high(Medium::Audio, "patterns"));
    }

    if audio_medium.is_match(name) {
        // For medium confidence, validate with file hints if available
        if hints.has_audio || (!hints.has_video && !hints.has_book && !hints.has_software) {
            return Some(StageResult::medium(Medium::Audio, "patterns"));
        }
    }

    None
}

// Software pattern checking
fn check_software_patterns(name: &str, hints: &FileHints) -> Option<StageResult> {
    static SOFTWARE_HIGH: OnceLock<Regex> = OnceLock::new();
    static SOFTWARE_MEDIUM: OnceLock<Regex> = OnceLock::new();

    // If we have video/audio files but no software files, skip software patterns
    if (hints.has_video || hints.has_audio) && !hints.has_software {
        return None;
    }

    // High confidence: crack indicators or known software vendors
    let software_high = SOFTWARE_HIGH.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Crack/activation indicators (very strong signal)
            \b(Crack|Keygen|Patch|Serial|Activator|Pre-?activated)\b
            |
            # Adobe/software patterns with product names
            \b(Adobe|Autodesk|JetBrains)\b.*?(CC|CS\d|Suite|\d{4})
            |
            # OS/distro ISO patterns
            \b(Windows|Ubuntu|Debian|Fedora|CentOS|Arch\s*Linux).*?\.(iso|img)\b
            |
            # Portable with version (strong software signal)
            \bv?\d+\.\d+(\.\d+)?.*?\bPortable\b
            |
            \bPortable\b.*?\bv?\d+\.\d+
            |
            # Repack with software context
            \bRepack\b.*?\b(x64|x86)\b
            "
        ).unwrap()
    });

    // Medium confidence: requires software file hints
    let software_medium = SOFTWARE_MEDIUM.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Version with platform (but not video codec versions)
            \bv?\d+\.\d+\.\d+\b.*?\b(x64|x86|Win|Mac|Linux|Portable)\b
            |
            \b(x64|x86|Win|Mac|Linux|Portable)\b.*?\bv?\d+\.\d+\.\d+\b
            "
        ).unwrap()
    });

    if software_high.is_match(name) {
        return Some(StageResult::high(Medium::Software, "patterns"));
    }

    if software_medium.is_match(name) {
        // Require software hints for medium confidence
        if hints.has_software || hints.has_archive {
            return Some(StageResult::medium(Medium::Software, "patterns"));
        }
    }

    None
}

// Book pattern checking
fn check_book_patterns(name: &str, hints: &FileHints) -> Option<StageResult> {
    static BOOK_HIGH: OnceLock<Regex> = OnceLock::new();
    static BOOK_MEDIUM: OnceLock<Regex> = OnceLock::new();

    // High confidence: explicit ebook format mentions
    let book_high = BOOK_HIGH.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Explicit format mentions
            \b(EPUB|MOBI|AZW3?|DJVU)\b
            |
            # Comic patterns
            \b(CBR|CBZ|CB7)\b
            |
            # Explicit ebook collection patterns
            \b(ebook|e-?book)\s*(collection|pack)\b
            "
        ).unwrap()
    });

    // Medium confidence: publisher names, ISBN
    let book_medium = BOOK_MEDIUM.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # ISBN pattern
            \bISBN\b
            |
            # Publisher names
            \b(O'?Reilly|Wiley|Springer|Packt|Manning|Apress|Penguin|HarperCollins)\b
            "
        ).unwrap()
    });

    if book_high.is_match(name) {
        return Some(StageResult::high(Medium::Book, "patterns"));
    }

    if book_medium.is_match(name) {
        // Validate with hints
        if hints.has_book || (!hints.has_video && !hints.has_audio && !hints.has_software) {
            return Some(StageResult::medium(Medium::Book, "patterns"));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::Confidence;

    fn classify_name(name: &str) -> Option<StageResult> {
        let stage = PatternStage::new();
        let info = ContentInfo {
            name: name.to_string(),
            files: vec![],
        };
        stage.classify(&info).unwrap()
    }

    // Video tests
    #[test]
    fn test_video_resolution_codec() {
        let result = classify_name("Movie.2024.1080p.BluRay.x264-GROUP");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Video);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_video_tv_show() {
        let result = classify_name("Show.Name.S01E05.720p.WEB-DL");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Video);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_video_4k() {
        let result = classify_name("Movie.2024.2160p.UHD.BluRay.HEVC");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Video);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_video_medium_confidence() {
        let result = classify_name("Some.Movie.720p");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Video);
        assert_eq!(r.confidence, Confidence::Medium);
    }

    // Audio tests
    #[test]
    fn test_audio_flac_album() {
        let result = classify_name("Artist - Album (2024) [FLAC]");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Audio);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_audio_mp3_bitrate() {
        let result = classify_name("Artist - Album (2024) 320kbps MP3");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Audio);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_audio_discography() {
        let result = classify_name("Artist - Discography (1990-2024)");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Audio);
    }

    #[test]
    fn test_audio_ost() {
        let result = classify_name("Movie Name - Original Soundtrack");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Audio);
    }

    // Software tests
    #[test]
    fn test_software_version_platform() {
        let result = classify_name("SomeApp v2.5.1 x64 Portable");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Software);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_software_crack() {
        let result = classify_name("SomeApp 2024 + Crack");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Software);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_software_adobe() {
        let result = classify_name("Adobe Photoshop CC 2024");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Software);
        assert_eq!(r.confidence, Confidence::High);
    }

    // Book tests
    #[test]
    fn test_book_epub() {
        let result = classify_name("Author - Book Title (EPUB)");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Book);
        assert_eq!(r.confidence, Confidence::High);
    }

    #[test]
    fn test_book_comic() {
        let result = classify_name("Comic Series Issue 1-50 (CBR)");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Book);
    }

    #[test]
    fn test_book_publisher() {
        let result = classify_name("O'Reilly - Learning Python");
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Book);
    }

    // Edge cases
    #[test]
    fn test_no_match() {
        let result = classify_name("random file name");
        assert!(result.is_none());
    }

    #[test]
    fn test_ambiguous_not_matched() {
        // Version number could be software, but alone isn't enough
        let result = classify_name("Something 2.0");
        // Should not match high confidence
        if let Some(r) = &result {
            assert_ne!(r.confidence, Confidence::High);
        }
    }
}
