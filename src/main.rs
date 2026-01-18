use flate2::read::GzDecoder;
use lava_torrent::torrent::v1::Torrent;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use tar::Archive as TarArchive;
use tokenizers::Tokenizer;
use xz2::read::XzDecoder;
use zip::ZipArchive;

const LABELS: [&str; 5] = ["audio", "video", "software", "book", "other"];
const MAX_LENGTH: usize = 128;

// Embed model files directly in the binary
const MODEL_BYTES: &[u8] =
    include_bytes!("../training/bert-classifier-medium/onnx/model_embedded.onnx");
const TOKENIZER_JSON: &str =
    include_str!("../training/bert-classifier-medium/onnx/tokenizer.json");

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

fn format_dir_for_classification(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let dir_name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Collect files with sizes
    let mut files: Vec<(u64, String)> = Vec::new();
    collect_files_recursive(path, &mut files)?;

    // Sort by size descending, take top 3
    files.sort_by(|a, b| b.0.cmp(&a.0));

    let mut lines = vec![dir_name];
    for (size, filename) in files.iter().take(3) {
        lines.push(format!("{} ({})", filename, human_size(*size)));
    }

    Ok(lines.join("\n"))
}

fn collect_files_recursive(
    path: &Path,
    files: &mut Vec<(u64, String)>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_file() {
            let size = entry.metadata()?.len();
            let name = entry.file_name().to_string_lossy().to_string();
            files.push((size, name));
        } else if file_type.is_dir() {
            collect_files_recursive(&entry.path(), files)?;
        }
    }
    Ok(())
}

fn format_zip_for_classification(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let archive_name = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut files: Vec<(u64, String)> = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        if !entry.is_dir() {
            let size = entry.size();
            let name = Path::new(entry.name())
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            files.push((size, name));
        }
    }

    // Sort by size descending, take top 3
    files.sort_by(|a, b| b.0.cmp(&a.0));

    let mut lines = vec![archive_name];
    for (size, filename) in files.iter().take(3) {
        lines.push(format!("{} ({})", filename, human_size(*size)));
    }

    Ok(lines.join("\n"))
}

fn is_zip_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("zip" | "cbz" | "epub" | "jar" | "apk")
    )
}

enum TarCompression {
    None,
    Gzip,
    Xz,
}

fn is_tar_file(path: &Path) -> Option<TarCompression> {
    let name = path.file_name()?.to_str()?;
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        Some(TarCompression::Gzip)
    } else if name.ends_with(".tar.xz") || name.ends_with(".txz") {
        Some(TarCompression::Xz)
    } else if name.ends_with(".tar") {
        Some(TarCompression::None)
    } else {
        None
    }
}

fn get_tar_stem(path: &Path) -> String {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    // Remove .tar.gz, .tar.xz, .tgz, .txz, or .tar
    let name = name
        .strip_suffix(".tar.gz")
        .or_else(|| name.strip_suffix(".tar.xz"))
        .or_else(|| name.strip_suffix(".tgz"))
        .or_else(|| name.strip_suffix(".txz"))
        .or_else(|| name.strip_suffix(".tar"))
        .unwrap_or(&name);
    name.to_string()
}

fn format_tar_for_classification(
    path: &Path,
    compression: TarCompression,
) -> Result<String, Box<dyn std::error::Error>> {
    let archive_name = get_tar_stem(path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut files: Vec<(u64, String)> = Vec::new();

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

    // Sort by size descending, take top 3
    files.sort_by(|a, b| b.0.cmp(&a.0));

    let mut lines = vec![archive_name];
    for (size, filename) in files.iter().take(3) {
        lines.push(format!("{} ({})", filename, human_size(*size)));
    }

    Ok(lines.join("\n"))
}

fn collect_tar_entries<R: std::io::Read>(
    archive: &mut TarArchive<R>,
    files: &mut Vec<(u64, String)>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in archive.entries()? {
        let entry = entry?;
        if entry.header().entry_type().is_file() {
            let size = entry.size();
            let name = entry
                .path()?
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            files.push((size, name));
        }
    }
    Ok(())
}

fn is_torrent_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("torrent"))
}

fn format_torrent_for_classification(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let torrent = Torrent::read_from_file(path)?;
    let name = torrent.name.clone();

    let mut files: Vec<(u64, String)> = Vec::new();

    match torrent.files {
        Some(file_list) => {
            // Multi-file torrent
            for file in file_list {
                let filename = file
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                files.push((file.length as u64, filename));
            }
        }
        None => {
            // Single-file torrent
            files.push((torrent.length as u64, name.clone()));
        }
    }

    // Sort by size descending, take top 3
    files.sort_by(|a, b| b.0.cmp(&a.0));

    let mut lines = vec![name];
    for (size, filename) in files.iter().take(3) {
        lines.push(format!("{} ({})", filename, human_size(*size)));
    }

    Ok(lines.join("\n"))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    logits.iter().map(|x| (x - max).exp() / exp_sum).collect()
}

struct MediumClassifier {
    session: Session,
    tokenizer: Tokenizer,
}

impl MediumClassifier {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Load session from embedded bytes
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(MODEL_BYTES)?;

        // Load tokenizer from embedded JSON
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes())
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self { session, tokenizer })
    }

    fn classify(&mut self, text: &str) -> Result<(&'static str, f32), Box<dyn std::error::Error>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization error: {}", e))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

        // Truncate if needed
        if input_ids.len() > MAX_LENGTH {
            input_ids.truncate(MAX_LENGTH);
            attention_mask.truncate(MAX_LENGTH);
        }

        let seq_len = input_ids.len();

        // Create tensors from ndarray
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Get input: from CLI args or stdin
    let raw_input = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        use std::io::{self, Read};
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer.trim().to_string()
    };

    if raw_input.is_empty() {
        eprintln!("Usage: mimmo_classifier <path|text>");
        eprintln!("   or: echo <text> | mimmo_classifier");
        eprintln!("");
        eprintln!("Supported inputs:");
        eprintln!("  - Directory path: classifies based on dir name + file contents");
        eprintln!("  - Zip file (.zip, .cbz, .epub, .jar, .apk)");
        eprintln!("  - Tar file (.tar, .tar.gz, .tgz, .tar.xz, .txz)");
        eprintln!("  - Torrent file (.torrent)");
        eprintln!("  - Raw text: classifies the text directly");
        std::process::exit(1);
    }

    // Check if input is a path
    let path = Path::new(&raw_input);
    let input = if path.is_dir() {
        format_dir_for_classification(path)?
    } else if path.is_file() && is_zip_file(path) {
        format_zip_for_classification(path)?
    } else if path.is_file() && is_torrent_file(path) {
        format_torrent_for_classification(path)?
    } else if path.is_file() {
        if let Some(compression) = is_tar_file(path) {
            format_tar_for_classification(path, compression)?
        } else {
            raw_input
        }
    } else {
        raw_input
    };

    let mut classifier = MediumClassifier::new()?;
    let (label, confidence) = classifier.classify(&input)?;

    println!(
        r#"{{"medium":"{}","confidence":{:.4}}}"#,
        label, confidence
    );

    Ok(())
}
