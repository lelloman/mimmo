//! Metadata extraction using fine-tuned SmolLM model
//!
//! Extracts structured metadata (title, artist, year) from torrent names
//! using a GGUF-quantized causal language model via llama.cpp.

use std::io::Write;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use crate::Error;

// Field counts for truncation
const FIELD_COUNTS: &[(&str, usize)] = &[
    ("video/movie", 2),    // title | year
    ("video/episode", 1),  // series_title
    ("video/season", 1),   // series_title
    ("video/series", 1),   // series_title
    ("audio/album", 3),    // album | artist | year
    ("audio/track", 3),    // track | artist | year
];

// Embed model file
const EXTRACTOR_MODEL: &[u8] = include_bytes!("../models/gguf/smollm-q4_k_m.gguf");

/// Extracted metadata from a torrent name
#[derive(Debug, Clone)]
pub struct ExtractedMetadata {
    /// For audio: album name. For video: title
    pub title: String,
    /// Artist name (audio only)
    pub artist: Option<String>,
    /// Release year
    pub year: Option<u16>,
    /// Raw model output before parsing
    pub raw_output: String,
}

/// Metadata extractor using SmolLM via llama.cpp
pub struct MetadataExtractor {
    model: LlamaModel,
    backend: LlamaBackend,
}

impl MetadataExtractor {
    /// Create a new metadata extractor
    pub fn new() -> Result<Self, Error> {
        // Initialize backend
        let backend = LlamaBackend::init()
            .map_err(|e| Error::Model(format!("Failed to init llama backend: {}", e)))?;

        // Write model to temp file (llama.cpp needs a file path)
        let mut temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| Error::Model(format!("Failed to create temp file: {}", e)))?;
        temp_file.write_all(EXTRACTOR_MODEL)
            .map_err(|e| Error::Model(format!("Failed to write model: {}", e)))?;
        let model_path = temp_file.into_temp_path();

        // Load model
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| Error::Model(format!("Failed to load model: {}", e)))?;

        Ok(Self { model, backend })
    }

    /// Extract metadata from a torrent name
    ///
    /// # Arguments
    /// * `name` - The torrent name
    /// * `content_type` - One of: "audio/album", "audio/track", "video/movie",
    ///                   "video/episode", "video/season", "video/series"
    pub fn extract(&self, name: &str, content_type: &str) -> Result<ExtractedMetadata, Error> {
        // Format prompt
        let prompt = format!(
            "<|im_start|>user\n<|extract|>[{}] {}<|im_end|>\n<|im_start|>assistant\n",
            content_type, name
        );

        // Generate response
        let raw_output = self.generate(&prompt, 32)?;

        // Parse based on content type
        let max_fields = FIELD_COUNTS
            .iter()
            .find(|(t, _)| *t == content_type)
            .map(|(_, c)| *c)
            .unwrap_or(3);

        let parsed = parse_output(&raw_output, max_fields, name);

        Ok(parsed)
    }

    /// Generate text using the model
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, Error> {
        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512));
        let mut ctx = self.model.new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Model(format!("Failed to create context: {}", e)))?;

        // Tokenize prompt
        let tokens = self.model.str_to_token(&prompt, llama_cpp_2::model::AddBos::Never)
            .map_err(|e| Error::Model(format!("Tokenization failed: {}", e)))?;

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(512, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| Error::Model(format!("Failed to add token to batch: {}", e)))?;
        }

        // Process prompt
        ctx.decode(&mut batch)
            .map_err(|e| Error::Model(format!("Decode failed: {}", e)))?;

        // Get EOS token
        let eos_token = self.model.token_eos();

        // Generate tokens
        let mut output_tokens = Vec::new();
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            // Get logits and sample
            let mut candidates = LlamaTokenDataArray::from_iter(
                ctx.candidates_ith(batch.n_tokens() - 1),
                false
            );

            // Greedy sampling (temperature 0)
            let token = candidates.sample_token_greedy();

            // Check for EOS
            if token == eos_token {
                break;
            }

            output_tokens.push(token);

            // Prepare next batch
            batch.clear();
            batch.add(token, n_cur as i32, &[0], true)
                .map_err(|e| Error::Model(format!("Failed to add token: {}", e)))?;

            ctx.decode(&mut batch)
                .map_err(|e| Error::Model(format!("Decode failed: {}", e)))?;

            n_cur += 1;
        }

        // Decode output tokens
        let output = output_tokens
            .iter()
            .filter_map(|t| self.model.token_to_str(*t, llama_cpp_2::model::Special::Tokenize).ok())
            .collect::<String>();

        Ok(output)
    }
}

/// Parse model output into structured metadata
fn parse_output(output: &str, max_fields: usize, input: &str) -> ExtractedMetadata {
    // Split by | and take only expected number of fields
    let fields: Vec<&str> = output
        .split('|')
        .take(max_fields)
        .map(|s| s.trim())
        .collect();

    // Validate year - must exist in input
    let validate_year = |s: &str| -> Option<u16> {
        if s.len() == 4 && s.chars().all(|c| c.is_ascii_digit()) {
            if input.contains(s) {
                s.parse().ok()
            } else {
                None // Hallucinated year
            }
        } else {
            None
        }
    };

    match max_fields {
        // video types: title | year or just title
        1 => ExtractedMetadata {
            title: fields.first().unwrap_or(&"").to_string(),
            artist: None,
            year: None,
            raw_output: output.to_string(),
        },
        2 => ExtractedMetadata {
            title: fields.first().unwrap_or(&"").to_string(),
            artist: None,
            year: fields.get(1).and_then(|s| validate_year(s)),
            raw_output: output.to_string(),
        },
        // audio types: title | artist | year
        3 => ExtractedMetadata {
            title: fields.first().unwrap_or(&"").to_string(),
            artist: fields.get(1).map(|s| s.to_string()),
            year: fields.get(2).and_then(|s| validate_year(s)),
            raw_output: output.to_string(),
        },
        _ => ExtractedMetadata {
            title: fields.first().unwrap_or(&"").to_string(),
            artist: None,
            year: None,
            raw_output: output.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_audio_album() {
        let output = "The Dark Side of the Moon | Pink Floyd | 1973";
        let input = "Pink Floyd - The Dark Side of the Moon (1973) [FLAC]";
        let parsed = parse_output(output, 3, input);

        assert_eq!(parsed.title, "The Dark Side of the Moon");
        assert_eq!(parsed.artist, Some("Pink Floyd".to_string()));
        assert_eq!(parsed.year, Some(1973));
    }

    #[test]
    fn test_parse_video_movie() {
        let output = "The Matrix | 1999";
        let input = "The.Matrix.1999.1080p.BluRay";
        let parsed = parse_output(output, 2, input);

        assert_eq!(parsed.title, "The Matrix");
        assert_eq!(parsed.year, Some(1999));
    }

    #[test]
    fn test_hallucinated_year_rejected() {
        let output = "Breaking Bad | 2023"; // 2023 not in input
        let input = "Breaking.Bad.S01E01.720p";
        let parsed = parse_output(output, 2, input);

        assert_eq!(parsed.title, "Breaking Bad");
        assert_eq!(parsed.year, None); // Rejected
    }

    #[test]
    fn test_extractor_audio_album() {
        let extractor = MetadataExtractor::new().expect("Failed to load model");
        let result = extractor.extract(
            "Pink Floyd - The Dark Side of the Moon (1973) [FLAC]",
            "audio/album"
        ).expect("Extraction failed");

        println!("Raw output: '{}'", result.raw_output);
        println!("Title: '{}'", result.title);
        println!("Artist: {:?}", result.artist);
        println!("Year: {:?}", result.year);

        assert!(!result.title.is_empty());
    }

    #[test]
    fn test_extractor_video_movie() {
        let extractor = MetadataExtractor::new().expect("Failed to load model");
        let result = extractor.extract(
            "The.Matrix.1999.1080p.BluRay.x264-GROUP",
            "video/movie"
        ).expect("Extraction failed");

        println!("Raw output: {}", result.raw_output);
        println!("Title: {}", result.title);
        println!("Year: {:?}", result.year);

        assert!(!result.title.is_empty());
    }
}
