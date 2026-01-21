use std::fs::{self, File};
use std::io;
use std::path::Path;

const MODEL_URL: &str = "https://huggingface.co/lelloman/smollm-torrent-metadata/resolve/main/smollm-q4_k_m.gguf";
const MODEL_PATH: &str = "models/gguf/smollm-q4_k_m.gguf";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", MODEL_PATH);

    let model_path = Path::new(MODEL_PATH);

    if model_path.exists() {
        println!("cargo:warning=Model already exists at {}", MODEL_PATH);
        return;
    }

    println!("cargo:warning=Downloading model from HuggingFace...");

    // Create directory if needed
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create models directory");
    }

    // Download the model
    let resp = ureq::get(MODEL_URL)
        .call()
        .expect("Failed to download model from HuggingFace");

    let mut file = File::create(model_path).expect("Failed to create model file");

    let mut reader = resp.into_reader();
    let bytes = io::copy(&mut reader, &mut file).expect("Failed to write model file");

    println!("cargo:warning=Downloaded {} bytes to {}", bytes, MODEL_PATH);
}
