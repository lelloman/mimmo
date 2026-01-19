use mimmo::cascade::{Cascade, Confidence, Medium, StageResult};
use mimmo::{from_path, from_text, Classifier, ClassificationResult};
use serde::Serialize;
use std::io::{self, BufRead, Write};
use std::path::Path;

#[derive(Serialize)]
struct JsonResult {
    medium: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    subcategory: Option<&'static str>,
    confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'static str>,
}

impl From<&ClassificationResult> for JsonResult {
    fn from(r: &ClassificationResult) -> Self {
        JsonResult {
            medium: r.medium,
            subcategory: r.subcategory,
            confidence: (r.confidence * 10000.0).round() / 10000.0,
            source: None,
        }
    }
}

impl From<&StageResult> for JsonResult {
    fn from(r: &StageResult) -> Self {
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
        JsonResult {
            medium,
            subcategory: None,
            confidence,
            source: Some(r.source),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Check for flags
    let use_cascade = args.iter().any(|a| a == "--cascade" || a == "-c");
    let interactive = args.iter().any(|a| a == "-i");

    if interactive {
        return interactive_mode(use_cascade);
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
        eprintln!("Usage: mimmo [--cascade|-c] <path|text>");
        eprintln!("   or: echo <text> | mimmo");
        eprintln!("   or: mimmo -i [--cascade]  (interactive JSON batch mode)");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -c, --cascade  Use cascade classifier (heuristics + ML fallback)");
        eprintln!("  -i             Interactive JSON batch mode");
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
    let json_result = if use_cascade {
        let cascade = Cascade::default_with_ml()?;
        let result = cascade.classify(&content_info)?;
        JsonResult::from(&result)
    } else {
        let mut classifier = Classifier::new()?;
        let result = classifier.classify(&content_info)?;
        JsonResult::from(&result)
    };

    // Output JSON
    println!("{}", serde_json::to_string(&json_result)?);

    Ok(())
}

enum ClassifierMode {
    Direct(Classifier),
    Cascade(Cascade),
}

fn interactive_mode(use_cascade: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mode = if use_cascade {
        ClassifierMode::Cascade(Cascade::default_with_ml()?)
    } else {
        ClassifierMode::Direct(Classifier::new()?)
    };

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

            let json_result = match &mode {
                ClassifierMode::Cascade(cascade) => {
                    let result = cascade.classify(&content_info)?;
                    JsonResult::from(&result)
                }
                ClassifierMode::Direct(classifier) => {
                    // Note: classify needs &mut self, but we don't have that here
                    // For now, use cascade in interactive mode with --cascade flag
                    let mut c = Classifier::new()?;
                    let result = c.classify(&content_info)?;
                    JsonResult::from(&result)
                }
            };
            results.push(json_result);
        }

        // Output JSON array of results
        println!("{}", serde_json::to_string(&results)?);
        stdout.flush()?;
    }

    Ok(())
}
