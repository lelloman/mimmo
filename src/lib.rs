//! Mimmo - Torrent content classifier
//!
//! A cascade-based classifier that predicts the media type of torrent content
//! using a combination of heuristics and ML fallback.
//!
//! # Architecture
//!
//! The classifier uses a cascade approach:
//! 1. Fast heuristic stages (file extensions, name patterns)
//! 2. ML fallback for ambiguous cases
//!
//! # Example
//!
//! ```no_run
//! use mimmo::{Classifier, from_torrent};
//!
//! let mut classifier = Classifier::new().unwrap();
//! let info = from_torrent("/path/to/file.torrent").unwrap();
//! let result = classifier.classify(&info).unwrap();
//!
//! println!("Medium: {}", result.medium);
//! println!("Confidence: {:.2}%", result.confidence * 100.0);
//! if let Some(sub) = result.subcategory {
//!     println!("Subcategory: {}", sub);
//! }
//! ```

use flate2::read::GzDecoder;
use lava_torrent::torrent::v1::Torrent;
use ndarray::Array2;
use once_cell::sync::Lazy;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use regex::Regex;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use tar::Archive as TarArchive;
use tokenizers::Tokenizer;
use xz2::read::XzDecoder;
use zip::ZipArchive;

pub use error::Error;

// Cascade classification system
pub mod cascade;

// Format extraction from torrent names
pub mod format;

// Metadata extraction using SmolLM
pub mod metadata;

mod error {
    use std::fmt;

    #[derive(Debug)]
    pub enum Error {
        Io(std::io::Error),
        Tokenizer(String),
        Torrent(String),
        Zip(zip::result::ZipError),
        Shape(ndarray::ShapeError),
        Ort(ort::Error),
        Model(String),
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Error::Io(e) => write!(f, "IO error: {}", e),
                Error::Tokenizer(e) => write!(f, "Tokenizer error: {}", e),
                Error::Torrent(e) => write!(f, "Torrent error: {}", e),
                Error::Zip(e) => write!(f, "Zip error: {}", e),
                Error::Shape(e) => write!(f, "Shape error: {}", e),
                Error::Ort(e) => write!(f, "ORT error: {}", e),
                Error::Model(e) => write!(f, "Model error: {}", e),
            }
        }
    }

    impl std::error::Error for Error {}

    impl From<std::io::Error> for Error {
        fn from(e: std::io::Error) -> Self {
            Error::Io(e)
        }
    }

    impl From<zip::result::ZipError> for Error {
        fn from(e: zip::result::ZipError) -> Self {
            Error::Zip(e)
        }
    }

    impl From<ndarray::ShapeError> for Error {
        fn from(e: ndarray::ShapeError) -> Self {
            Error::Shape(e)
        }
    }

    impl From<ort::Error> for Error {
        fn from(e: ort::Error) -> Self {
            Error::Ort(e)
        }
    }

    impl From<lava_torrent::LavaTorrentError> for Error {
        fn from(e: lava_torrent::LavaTorrentError) -> Self {
            Error::Torrent(e.to_string())
        }
    }
}

const LABELS: [&str; 5] = ["audio", "video", "software", "book", "other"];
const MAX_LENGTH: usize = 128;

// Embed model files directly in the binary
const MODEL_BYTES: &[u8] = include_bytes!("../models/bert/model_embedded.onnx");
const TOKENIZER_JSON: &str = include_str!("../models/bert/tokenizer.json");

// Regex pattern for episode detection (S01E01, 1x01, etc)
static EPISODE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)[sS](\d{1,2})[eE](\d{1,3})|(\d{1,2})x(\d{1,2})").unwrap()
});

/// Information about a single file within the content
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Full path within archive/directory
    pub path: String,
    /// Just the filename
    pub filename: String,
    /// File size in bytes
    pub size: u64,
}

/// Collected content info for classification
#[derive(Debug)]
pub struct ContentInfo {
    /// Name of the content (torrent name, directory name, etc.)
    pub name: String,
    /// List of files within the content
    pub files: Vec<FileInfo>,
}

/// Classification result
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Primary medium type: "audio", "video", "software", "book", or "other"
    pub medium: &'static str,
    /// Model confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Subcategory for audio/video content
    /// - audio: "album", "collection", "track", or "other"
    /// - video: "movie", "series", "season", "episode", or "other"
    pub subcategory: Option<&'static str>,
}

/// The main classifier
pub struct Classifier {
    /// ONNX inference session
    pub session: Session,
    /// Tokenizer for text encoding
    pub tokenizer: Tokenizer,
}

impl Classifier {
    /// Create a new classifier instance
    pub fn new() -> Result<Self, Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(MODEL_BYTES)?;

        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    /// Classify content and return medium, confidence, and subcategory
    pub fn classify(&mut self, info: &ContentInfo) -> Result<ClassificationResult, Error> {
        let input_text = format_for_classification(info);
        let (medium, confidence) = self.classify_text(&input_text)?;

        let subcategory = if medium == "audio" || medium == "video" {
            Some(detect_subcategory(medium, info))
        } else {
            None
        };

        Ok(ClassificationResult {
            medium,
            confidence,
            subcategory,
        })
    }

    /// Classify raw text (without file structure info)
    pub fn classify_text(&mut self, text: &str) -> Result<(&'static str, f32), Error> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

        if input_ids.len() > MAX_LENGTH {
            input_ids.truncate(MAX_LENGTH);
            attention_mask.truncate(MAX_LENGTH);
        }

        let seq_len = input_ids.len();

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)?;

        let input_ids_tensor = Tensor::from_array(input_ids_arr)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_arr)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor
        ])?;

        let logits_arr = outputs["logits"].try_extract_array::<f32>()?;
        let logits_slice: Vec<f32> = logits_arr.iter().cloned().collect();
        let probs = softmax(&logits_slice);

        let (max_idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((LABELS[max_idx], max_prob))
    }
}

// =============================================================================
// CONTENT LOADERS
// =============================================================================

/// Load content info from a torrent file
pub fn from_torrent<P: AsRef<Path>>(path: P) -> Result<ContentInfo, Error> {
    let path = path.as_ref();
    let torrent = Torrent::read_from_file(path)?;
    let name = torrent.name.clone();

    let mut files = Vec::new();

    match torrent.files {
        Some(file_list) => {
            for file in file_list {
                let entry_path = file.path.to_string_lossy().to_string();
                let filename = file
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                files.push(FileInfo {
                    path: entry_path,
                    filename,
                    size: file.length as u64,
                });
            }
        }
        None => {
            // Single-file torrent
            files.push(FileInfo {
                path: name.clone(),
                filename: name.clone(),
                size: torrent.length as u64,
            });
        }
    }

    Ok(ContentInfo { name, files })
}

/// Load content info from a directory
pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<ContentInfo, Error> {
    let path = path.as_ref();
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut files = Vec::new();
    collect_files_recursive(path, path, &mut files)?;

    Ok(ContentInfo { name, files })
}

/// Load content info from a zip file
pub fn from_zip<P: AsRef<Path>>(path: P) -> Result<ContentInfo, Error> {
    let path = path.as_ref();
    let name = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut files = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        if !entry.is_dir() {
            let entry_path = entry.name().to_string();
            let filename = Path::new(&entry_path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            files.push(FileInfo {
                path: entry_path,
                filename,
                size: entry.size(),
            });
        }
    }

    Ok(ContentInfo { name, files })
}

/// Load content info from a tar archive (uncompressed, gzip, or xz)
pub fn from_tar<P: AsRef<Path>>(path: P) -> Result<ContentInfo, Error> {
    let path = path.as_ref();
    let compression = detect_tar_compression(path);
    let name = get_tar_stem(path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut files = Vec::new();

    match compression {
        TarCompression::None => {
            let mut archive = TarArchive::new(reader);
            collect_tar_entries(&mut archive, &mut files)?;
        }
        TarCompression::Gzip => {
            let decoder = GzDecoder::new(reader);
            let mut archive = TarArchive::new(decoder);
            collect_tar_entries(&mut archive, &mut files)?;
        }
        TarCompression::Xz => {
            let decoder = XzDecoder::new(reader);
            let mut archive = TarArchive::new(decoder);
            collect_tar_entries(&mut archive, &mut files)?;
        }
    }

    Ok(ContentInfo { name, files })
}

/// Create content info from raw text (no file structure)
pub fn from_text(text: &str) -> ContentInfo {
    ContentInfo {
        name: text.to_string(),
        files: Vec::new(),
    }
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

fn human_size(n: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = n as f64;
    for unit in units {
        if size < 1024.0 {
            return format!("{:.1}{}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1}PB", size)
}

fn is_video_file(filename: &str) -> bool {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(
        ext.as_str(),
        "mkv" | "mp4" | "avi" | "wmv" | "mov" | "m4v" | "ts" | "m2ts"
    )
}

fn is_audio_file(filename: &str) -> bool {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(
        ext.as_str(),
        "mp3" | "flac" | "m4a" | "ogg" | "opus" | "wav" | "aac" | "wma"
    )
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    logits.iter().map(|x| (x - max).exp() / exp_sum).collect()
}

fn format_for_classification(info: &ContentInfo) -> String {
    let mut sorted_files = info.files.clone();
    sorted_files.sort_by(|a, b| b.size.cmp(&a.size));

    let mut lines = vec![info.name.clone()];
    for file in sorted_files.iter().take(3) {
        lines.push(format!("{} ({})", file.filename, human_size(file.size)));
    }

    lines.join("\n")
}

// =============================================================================
// SUBCATEGORY DETECTION
// =============================================================================

struct MediaStructure {
    media_file_count: usize,
    dir_count: usize,
    has_episode_pattern: bool,
}

fn analyze_structure(info: &ContentInfo, is_video: bool) -> MediaStructure {
    let media_files: Vec<&FileInfo> = info
        .files
        .iter()
        .filter(|f| {
            if is_video {
                is_video_file(&f.filename)
            } else {
                is_audio_file(&f.filename)
            }
        })
        .collect();

    let media_file_count = media_files.len();

    let unique_dirs: HashSet<String> = media_files
        .iter()
        .map(|f| match f.path.rfind('/') {
            Some(pos) => f.path[..pos].to_string(),
            None => String::new(),
        })
        .collect();
    let dir_count = unique_dirs.len();

    let has_episode_pattern = is_video
        && media_files
            .iter()
            .any(|f| EPISODE_PATTERN.is_match(&f.filename) || EPISODE_PATTERN.is_match(&f.path));

    MediaStructure {
        media_file_count,
        dir_count,
        has_episode_pattern,
    }
}

/// Detect subcategory for audio/video content based on file structure.
pub fn detect_subcategory(medium: &str, info: &ContentInfo) -> &'static str {
    let is_video = medium == "video";
    let structure = analyze_structure(info, is_video);

    if structure.media_file_count == 0 {
        return "other";
    }

    match (structure.media_file_count, structure.dir_count) {
        // Single file
        (1, _) => {
            if is_video {
                if structure.has_episode_pattern {
                    "episode"
                } else {
                    "movie"
                }
            } else {
                "track"
            }
        }
        // Multiple files in single directory
        (_, 1) => {
            if is_video {
                "season"
            } else {
                "album"
            }
        }
        // Multiple files in multiple directories
        (_, 2..) => {
            if is_video {
                "series"
            } else {
                "collection"
            }
        }
        // Fallback
        _ => "other",
    }
}

// =============================================================================
// TAR HELPERS
// =============================================================================

enum TarCompression {
    None,
    Gzip,
    Xz,
}

fn detect_tar_compression(path: &Path) -> TarCompression {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        TarCompression::Gzip
    } else if name.ends_with(".tar.xz") || name.ends_with(".txz") {
        TarCompression::Xz
    } else {
        TarCompression::None
    }
}

fn get_tar_stem(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let name = name
        .strip_suffix(".tar.gz")
        .or_else(|| name.strip_suffix(".tar.xz"))
        .or_else(|| name.strip_suffix(".tgz"))
        .or_else(|| name.strip_suffix(".txz"))
        .or_else(|| name.strip_suffix(".tar"))
        .unwrap_or(&name);
    name.to_string()
}

fn collect_files_recursive(
    base: &Path,
    current: &Path,
    files: &mut Vec<FileInfo>,
) -> Result<(), Error> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let entry_path = entry.path();

        let rel_path = entry_path
            .strip_prefix(base)
            .unwrap_or(&entry_path)
            .to_string_lossy()
            .to_string();

        if file_type.is_file() {
            let size = entry.metadata()?.len();
            let filename = entry.file_name().to_string_lossy().to_string();
            files.push(FileInfo {
                path: rel_path,
                filename,
                size,
            });
        } else if file_type.is_dir() {
            collect_files_recursive(base, &entry_path, files)?;
        }
    }
    Ok(())
}

fn collect_tar_entries<R: std::io::Read>(
    archive: &mut TarArchive<R>,
    files: &mut Vec<FileInfo>,
) -> Result<(), Error> {
    for entry in archive.entries()? {
        let entry = entry?;
        if entry.header().entry_type().is_file() {
            let entry_path = entry.path()?.to_string_lossy().to_string();
            let filename = Path::new(&entry_path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            files.push(FileInfo {
                path: entry_path,
                filename,
                size: entry.size(),
            });
        }
    }
    Ok(())
}

// =============================================================================
// FILE TYPE DETECTION (for auto-loading)
// =============================================================================

/// Check if path is a zip-like file
pub fn is_zip_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("zip" | "cbz" | "epub" | "jar" | "apk")
    )
}

/// Check if path is a tar file (returns true for .tar, .tar.gz, .tar.xz, etc.)
pub fn is_tar_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.ends_with(".tar.gz")
        || name.ends_with(".tgz")
        || name.ends_with(".tar.xz")
        || name.ends_with(".txz")
        || name.ends_with(".tar")
}

/// Check if path is a torrent file
pub fn is_torrent_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("torrent"))
}

/// Auto-detect file type and load content info
pub fn from_path<P: AsRef<Path>>(path: P) -> Result<ContentInfo, Error> {
    let path = path.as_ref();

    if path.is_dir() {
        from_directory(path)
    } else if is_torrent_file(path) {
        from_torrent(path)
    } else if is_zip_file(path) {
        from_zip(path)
    } else if is_tar_file(path) {
        from_tar(path)
    } else {
        // Treat as raw text (filename)
        Ok(from_text(&path.to_string_lossy()))
    }
}
