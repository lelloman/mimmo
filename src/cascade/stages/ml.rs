//! Machine learning fallback stage.
//!
//! This stage uses the BERT-based classifier as a fallback when heuristic
//! stages cannot confidently classify the content.

use std::sync::Mutex;

use crate::cascade::{Confidence, Medium, Stage, StageResult};
use crate::{Classifier, ContentInfo, Error};

/// Stage that uses the BERT classifier as a fallback.
///
/// This stage should typically be placed last in the cascade, after all
/// heuristic stages have had a chance to classify with high confidence.
///
/// Confidence mapping:
/// - >= 0.9: High confidence
/// - >= 0.7: Medium confidence
/// - < 0.7: Low confidence (still returns a result)
pub struct MlStage {
    classifier: Mutex<Classifier>,
    /// Minimum confidence to return a result (otherwise pass to next stage)
    min_confidence: f32,
}

impl MlStage {
    /// Create a new ML stage with the embedded BERT model.
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            classifier: Mutex::new(Classifier::new()?),
            min_confidence: 0.0, // Always return a result by default
        })
    }

    /// Set minimum confidence threshold.
    ///
    /// If the ML model's confidence is below this threshold, the stage
    /// will return `None` to pass to the next stage (or cascade fallback).
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold.clamp(0.0, 1.0);
        self
    }

    /// Map string label to Medium enum
    fn label_to_medium(label: &str) -> Medium {
        match label {
            "video" => Medium::Video,
            "audio" => Medium::Audio,
            "book" => Medium::Book,
            "software" => Medium::Software,
            _ => Medium::Other,
        }
    }

    /// Map confidence score to confidence level
    fn score_to_confidence(score: f32) -> Confidence {
        if score >= 0.9 {
            Confidence::High
        } else if score >= 0.7 {
            Confidence::Medium
        } else {
            Confidence::Low
        }
    }
}

impl Stage for MlStage {
    fn name(&self) -> &'static str {
        "ml"
    }

    fn classify(&self, info: &ContentInfo) -> Result<Option<StageResult>, Error> {
        let input_text = format_for_classification(info);

        // Lock the mutex to get mutable access to the classifier
        let mut classifier = self
            .classifier
            .lock()
            .map_err(|_| Error::Tokenizer("Mutex poisoned".to_string()))?;

        let (label, score) = classify_with_session(&mut classifier, &input_text)?;

        // Check minimum confidence threshold
        if score < self.min_confidence {
            return Ok(None);
        }

        let medium = Self::label_to_medium(label);
        let confidence = Self::score_to_confidence(score);

        Ok(Some(StageResult {
            medium,
            confidence,
            source: "ml",
        }))
    }
}

/// Format content info for classification (same as in lib.rs)
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

/// Classify text using the ONNX session
fn classify_with_session(classifier: &mut Classifier, text: &str) -> Result<(&'static str, f32), Error> {
    use ndarray::Array2;
    use ort::value::Tensor;

    const LABELS: [&str; 5] = ["audio", "video", "software", "book", "other"];
    const MAX_LENGTH: usize = 128;

    let encoding = classifier
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

    let outputs = classifier.session.run(ort::inputs![
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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    logits.iter().map(|x| (x - max).exp() / exp_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_to_medium() {
        assert_eq!(MlStage::label_to_medium("video"), Medium::Video);
        assert_eq!(MlStage::label_to_medium("audio"), Medium::Audio);
        assert_eq!(MlStage::label_to_medium("book"), Medium::Book);
        assert_eq!(MlStage::label_to_medium("software"), Medium::Software);
        assert_eq!(MlStage::label_to_medium("other"), Medium::Other);
        assert_eq!(MlStage::label_to_medium("unknown"), Medium::Other);
    }

    #[test]
    fn test_score_to_confidence() {
        assert_eq!(MlStage::score_to_confidence(0.95), Confidence::High);
        assert_eq!(MlStage::score_to_confidence(0.90), Confidence::High);
        assert_eq!(MlStage::score_to_confidence(0.85), Confidence::Medium);
        assert_eq!(MlStage::score_to_confidence(0.70), Confidence::Medium);
        assert_eq!(MlStage::score_to_confidence(0.69), Confidence::Low);
        assert_eq!(MlStage::score_to_confidence(0.50), Confidence::Low);
    }

    #[test]
    fn test_human_size() {
        assert_eq!(human_size(100), "100.0B");
        assert_eq!(human_size(1024), "1.0KB");
        assert_eq!(human_size(1_048_576), "1.0MB");
        assert_eq!(human_size(1_073_741_824), "1.0GB");
    }

    // Integration test - requires model to be available
    #[test]
    fn test_ml_stage_creation() {
        let stage = MlStage::new();
        assert!(stage.is_ok());
        let stage = stage.unwrap();
        assert_eq!(stage.name(), "ml");
    }

    #[test]
    fn test_ml_stage_classifies_video() {
        let stage = MlStage::new().unwrap();
        let info = ContentInfo {
            name: "Movie.2024.1080p.BluRay.x264-GROUP".to_string(),
            files: vec![crate::FileInfo {
                path: "movie.mkv".to_string(),
                filename: "movie.mkv".to_string(),
                size: 4_000_000_000,
            }],
        };

        let result = stage.classify(&info).unwrap();
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.medium, Medium::Video);
    }
}
