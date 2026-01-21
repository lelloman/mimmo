use mimmo::cascade::{Cascade, Confidence, Medium, StageResult};
use mimmo::metadata::MetadataExtractor;
use mimmo::{from_path, from_text, detect_subcategory, ContentInfo};
use serde::Serialize;
use std::io::{self, BufRead, Write};
use std::path::Path;

#[derive(Serialize)]
struct ExtractedMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    artist: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    year: Option<u16>,
}

#[derive(Serialize)]
struct JsonResult {
    medium: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    subcategory: Option<&'static str>,
    confidence: f32,
    source: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<ExtractedMetadata>,
}

fn stage_result_to_json(r: &StageResult, info: &ContentInfo, extractor: Option<&MetadataExtractor>) -> JsonResult {
    let confidence = match r.confidence {
        Confidence::High => 0.95,
        Confidence::Medium => 0.75,
        Confidence::Low => 0.50,
    };
    let medium = match r.medium {
        Medium::Video => "video",
        Medium::Audio => "audio",
        Medium::Book => "book",
        Medium::Software => "software",
        Medium::Other => "other",
    };
    let subcategory = if medium == "audio" || medium == "video" {
        Some(detect_subcategory(medium, info))
    } else {
        None
    };

    // Extract metadata for audio/video content
    let metadata = if let (Some(ext), Some(sub)) = (extractor, subcategory) {
        let content_type = format!("{}/{}", medium, sub);
        match ext.extract(&info.name, &content_type) {
            Ok(m) => Some(ExtractedMetadata {
                title: if m.title.is_empty() { None } else { Some(m.title) },
                artist: m.artist,
                year: m.year,
            }),
            Err(_) => None,
        }
    } else {
        None
    };

    JsonResult {
        medium,
        subcategory,
        confidence,
        source: r.source,
        metadata,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let interactive = args.iter().any(|a| a == "-i");

    if interactive {
        return interactive_mode();
    }

    // Get non-flag arguments
    let input_args: Vec<&String> = args[1..]
        .iter()
        .filter(|a| !a.starts_with('-'))
        .collect();

    let raw_input = if !input_args.is_empty() {
        input_args
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        use std::io::Read;
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer.trim().to_string()
    };

    if raw_input.is_empty() {
        eprintln!("Usage: mimmo <path|text>");
        eprintln!("   or: echo <text> | mimmo");
        eprintln!("   or: mimmo -i  (interactive JSON batch mode)");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -i  Interactive JSON batch mode");
        eprintln!();
        eprintln!("Supported inputs:");
        eprintln!("  - Directory path");
        eprintln!("  - Zip file (.zip, .cbz, .epub, .jar, .apk)");
        eprintln!("  - Tar file (.tar, .tar.gz, .tgz, .tar.xz, .txz)");
        eprintln!("  - Torrent file (.torrent)");
        eprintln!("  - Raw text");
        eprintln!();
        eprintln!("Output: JSON with medium, subcategory, confidence, and source");
        std::process::exit(1);
    }

    let path = Path::new(&raw_input);

    // Load content info
    let content_info = if path.exists() {
        from_path(path)?
    } else {
        from_text(&raw_input)
    };

    // Classify
    let cascade = Cascade::default_with_ml()?;
    let result = cascade.classify(&content_info)?;

    // Create metadata extractor for audio/video
    let extractor = if matches!(result.medium, Medium::Audio | Medium::Video) {
        MetadataExtractor::new().ok()
    } else {
        None
    };

    let json_result = stage_result_to_json(&result, &content_info, extractor.as_ref());

    // Output JSON
    println!("{}", serde_json::to_string(&json_result)?);

    Ok(())
}

fn interactive_mode() -> Result<(), Box<dyn std::error::Error>> {
    let cascade = Cascade::default_with_ml()?;
    let extractor = MetadataExtractor::new().ok();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        // Parse JSON array of input strings
        let inputs: Vec<String> = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("JSON parse error: {}", e);
                continue;
            }
        };

        // Classify each input
        let mut results: Vec<JsonResult> = Vec::with_capacity(inputs.len());
        for input in &inputs {
            let path = Path::new(input);
            let content_info = if path.exists() {
                from_path(path)?
            } else {
                from_text(input)
            };

            let result = cascade.classify(&content_info)?;
            let ext_ref = if matches!(result.medium, Medium::Audio | Medium::Video) {
                extractor.as_ref()
            } else {
                None
            };
            results.push(stage_result_to_json(&result, &content_info, ext_ref));
        }

        // Output JSON array of results
        println!("{}", serde_json::to_string(&results)?);
        stdout.flush()?;
    }

    Ok(())
}
