use mimmo::{from_path, from_text, Classifier, ClassificationResult};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::path::Path;

#[derive(Serialize)]
struct JsonResult {
    medium: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    subcategory: Option<&'static str>,
    confidence: f32,
}

impl From<&ClassificationResult> for JsonResult {
    fn from(r: &ClassificationResult) -> Self {
        JsonResult {
            medium: r.medium,
            subcategory: r.subcategory,
            confidence: (r.confidence * 10000.0).round() / 10000.0,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Check for interactive mode
    if args.len() > 1 && args[1] == "-i" {
        return interactive_mode();
    }

    let raw_input = if args.len() > 1 {
        args[1..].join(" ")
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
        eprintln!("Supported inputs:");
        eprintln!("  - Directory path");
        eprintln!("  - Zip file (.zip, .cbz, .epub, .jar, .apk)");
        eprintln!("  - Tar file (.tar, .tar.gz, .tgz, .tar.xz, .txz)");
        eprintln!("  - Torrent file (.torrent)");
        eprintln!("  - Raw text");
        eprintln!();
        eprintln!("Output: JSON with medium, subcategory, and confidence");
        eprintln!();
        eprintln!("Interactive mode (-i):");
        eprintln!("  Input:  JSON array of strings, one per line");
        eprintln!("          [\"torrent name 1\", \"torrent name 2\", ...]");
        eprintln!("  Output: JSON array of results");
        eprintln!("          [{{\"medium\":\"video\",...}}, ...]");
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
    let mut classifier = Classifier::new()?;
    let result = classifier.classify(&content_info)?;

    // Output JSON
    println!("{}", serde_json::to_string(&JsonResult::from(&result))?);

    Ok(())
}

fn interactive_mode() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::new()?;
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

            let result = classifier.classify(&content_info)?;
            results.push(JsonResult::from(&result));
        }

        // Output JSON array of results
        println!("{}", serde_json::to_string(&results)?);
        stdout.flush()?;
    }

    Ok(())
}
