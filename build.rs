use std::fs::{self, File};
use std::io;
use std::path::Path;

const HF_BASE_SMOLLM: &str = "https://huggingface.co/lelloman/smollm-torrent-metadata/resolve/main";
const HF_BASE_BERT: &str = "https://huggingface.co/lelloman/bert-torrent-classifier/resolve/main";

struct ModelFile {
    url: String,
    path: &'static str,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let models = [
        ModelFile {
            url: format!("{}/smollm-q4_k_m.gguf", HF_BASE_SMOLLM),
            path: "models/gguf/smollm-q4_k_m.gguf",
        },
        ModelFile {
            url: format!("{}/model_embedded.onnx", HF_BASE_BERT),
            path: "models/bert/model_embedded.onnx",
        },
        ModelFile {
            url: format!("{}/tokenizer.json", HF_BASE_BERT),
            path: "models/bert/tokenizer.json",
        },
    ];

    for model in &models {
        println!("cargo:rerun-if-changed={}", model.path);
        download_if_missing(&model.url, model.path);
    }
}

fn download_if_missing(url: &str, path: &str) {
    let model_path = Path::new(path);

    if model_path.exists() {
        println!("cargo:warning=Model already exists at {}", path);
        return;
    }

    println!("cargo:warning=Downloading {} from HuggingFace...", path);

    // Create directory if needed
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create models directory");
    }

    // Download the model
    let resp = ureq::get(url)
        .call()
        .expect("Failed to download model from HuggingFace");

    let mut file = File::create(model_path).expect("Failed to create model file");

    let mut reader = resp.into_reader();
    let bytes = io::copy(&mut reader, &mut file).expect("Failed to write model file");

    println!("cargo:warning=Downloaded {} bytes to {}", bytes, path);
}
