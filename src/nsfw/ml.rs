//! Machine learning NSFW detection stage.
//!
//! Uses a BERT-tiny binary classifier for NSFW detection.
//! This is the final stage in the NSFW cascade and always returns a result.

use std::sync::{Arc, Mutex};

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::nsfw::{NsfwResult, NsfwSource, NsfwStage};
use crate::{get_shared_tokenizer, ContentInfo, Error, NSFW_MODEL_BYTES};

const MAX_LENGTH: usize = 128;

/// Stage that uses a BERT-tiny binary classifier for NSFW detection.
///
/// This stage always returns a result, making it suitable as the final
/// stage in the NSFW cascade.
pub struct NsfwMlStage {
    session: Mutex<Session>,
    tokenizer: Arc<Tokenizer>,
}

impl NsfwMlStage {
    /// Create a new NSFW ML stage with the embedded model.
    pub fn new() -> Result<Self, Error> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(NSFW_MODEL_BYTES)?;

        let tokenizer = get_shared_tokenizer()?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Classify text and return (is_nsfw, confidence)
    fn classify_text(&self, text: &str) -> Result<(bool, f32), Error> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        if input_ids.len() > MAX_LENGTH {
            input_ids.truncate(MAX_LENGTH);
            attention_mask.truncate(MAX_LENGTH);
        }

        let seq_len = input_ids.len();

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)?;

        let input_ids_tensor = Tensor::from_array(input_ids_arr)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_arr)?;

        // Lock the mutex to get mutable access to the session
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::Tokenizer("Mutex poisoned".to_string()))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor
        ])?;

        let logits_arr = outputs["logits"].try_extract_array::<f32>()?;
        let logits_slice: Vec<f32> = logits_arr.iter().cloned().collect();
        let probs = softmax(&logits_slice);

        // Binary classification: index 0 = not nsfw, index 1 = nsfw
        // Return the prediction and confidence for that prediction
        let (is_nsfw, confidence) = if probs.len() >= 2 {
            if probs[1] > probs[0] {
                (true, probs[1])
            } else {
                (false, probs[0])
            }
        } else if !probs.is_empty() {
            // Single output (sigmoid style)
            let p = probs[0];
            if p > 0.5 {
                (true, p)
            } else {
                (false, 1.0 - p)
            }
        } else {
            (false, 0.5)
        };

        Ok((is_nsfw, confidence))
    }
}

impl NsfwStage for NsfwMlStage {
    fn name(&self) -> &'static str {
        "nsfw_ml"
    }

    fn classify(&self, info: &ContentInfo) -> Result<Option<NsfwResult>, Error> {
        let input_text = format_for_classification(info);
        let (is_nsfw, confidence) = self.classify_text(&input_text)?;

        // ML stage always returns a result
        Ok(Some(NsfwResult::new(is_nsfw, confidence, NsfwSource::Ml)))
    }
}

/// Format content info for classification.
fn format_for_classification(info: &ContentInfo) -> String {
    let mut sorted_files = info.files.clone();
    sorted_files.sort_by(|a, b| b.size.cmp(&a.size));

    let mut lines = vec![info.name.clone()];
    for file in sorted_files.iter().take(3) {
        lines.push(format!("{} ({})", file.filename, human_size(file.size)));
    }

    lines.join("\n")
}

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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    logits.iter().map(|x| (x - max).exp() / exp_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsfw_ml_stage_creation() {
        let stage = NsfwMlStage::new();
        assert!(stage.is_ok());
        let stage = stage.unwrap();
        assert_eq!(stage.name(), "nsfw_ml");
    }

    #[test]
    fn test_nsfw_ml_classifies_safe_content() {
        let stage = NsfwMlStage::new().unwrap();
        let info = ContentInfo {
            name: "The.Matrix.1999.1080p.BluRay.x264".to_string(),
            files: vec![],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_some());
        let r = result.unwrap();
        // Should classify as not NSFW
        assert!(!r.nsfw);
        assert_eq!(r.source, NsfwSource::Ml);
    }

    #[test]
    fn test_nsfw_ml_always_returns_result() {
        let stage = NsfwMlStage::new().unwrap();
        let info = ContentInfo {
            name: "random content".to_string(),
            files: vec![],
        };

        let result = stage.classify(&info).unwrap();
        // ML stage should always return Some
        assert!(result.is_some());
    }

    #[test]
    fn test_human_size() {
        assert_eq!(human_size(100), "100.0B");
        assert_eq!(human_size(1024), "1.0KB");
        assert_eq!(human_size(1_048_576), "1.0MB");
        assert_eq!(human_size(1_073_741_824), "1.0GB");
    }
}
