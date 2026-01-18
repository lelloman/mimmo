use mimmo::{from_path, from_text, Classifier};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let raw_input = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        use std::io::{self, Read};
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer.trim().to_string()
    };

    if raw_input.is_empty() {
        eprintln!("Usage: mimmo <path|text>");
        eprintln!("   or: echo <text> | mimmo");
        eprintln!();
        eprintln!("Supported inputs:");
        eprintln!("  - Directory path");
        eprintln!("  - Zip file (.zip, .cbz, .epub, .jar, .apk)");
        eprintln!("  - Tar file (.tar, .tar.gz, .tgz, .tar.xz, .txz)");
        eprintln!("  - Torrent file (.torrent)");
        eprintln!("  - Raw text");
        eprintln!();
        eprintln!("Output: JSON with medium, subcategory, and confidence");
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
    match result.subcategory {
        Some(sub) => {
            println!(
                r#"{{"medium":"{}","subcategory":"{}","confidence":{:.4}}}"#,
                result.medium, sub, result.confidence
            );
        }
        None => {
            println!(
                r#"{{"medium":"{}","confidence":{:.4}}}"#,
                result.medium, result.confidence
            );
        }
    }

    Ok(())
}
