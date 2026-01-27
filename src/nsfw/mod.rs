//! NSFW detection cascade for torrent content.
//!
//! This module provides NSFW classification that runs orthogonally to
//! the medium classification. A torrent can be `video/movie` AND `nsfw: true`.
//!
//! # Architecture
//!
//! The NSFW cascade uses three stages:
//! 1. **Keywords** - HashSet lookup for explicit terms (O(1), ~0.1ms)
//! 2. **Patterns** - Regex for JAV codes, studio patterns (~1ms)
//! 3. **ML** - BERT-tiny binary classifier (~10ms)

mod keywords;
mod ml;
mod patterns;

pub use keywords::NsfwKeywordStage;
pub use ml::NsfwMlStage;
pub use patterns::NsfwPatternStage;

use crate::{ContentInfo, Error};

/// Source of the NSFW classification decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NsfwSource {
    /// Matched explicit keywords in name or filenames
    Keywords,
    /// Matched NSFW-specific patterns (JAV codes, studio patterns)
    Patterns,
    /// ML classifier decision
    Ml,
}

impl NsfwSource {
    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            NsfwSource::Keywords => "keywords",
            NsfwSource::Patterns => "patterns",
            NsfwSource::Ml => "ml",
        }
    }
}

/// Result from NSFW classification.
#[derive(Debug, Clone)]
pub struct NsfwResult {
    /// Whether the content is NSFW
    pub nsfw: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Source of the classification
    pub source: NsfwSource,
}

impl NsfwResult {
    /// Create a new NSFW result.
    pub fn new(nsfw: bool, confidence: f32, source: NsfwSource) -> Self {
        Self {
            nsfw,
            confidence,
            source,
        }
    }
}

/// A single stage in the NSFW classification cascade.
///
/// Each stage examines the content and either:
/// - Returns `Some(result)` if it can confidently classify as NSFW
/// - Returns `None` to pass to the next stage
pub trait NsfwStage: Send + Sync {
    /// The name of this stage (for debugging/logging).
    fn name(&self) -> &'static str;

    /// Try to classify the content as NSFW.
    ///
    /// Returns:
    /// - `Ok(Some(result))` if classification is confident
    /// - `Ok(None)` to pass to the next stage
    /// - `Err(e)` on error (will propagate up)
    fn classify(&self, info: &ContentInfo) -> Result<Option<NsfwResult>, Error>;
}

/// NSFW classification cascade.
///
/// Runs stages in order: Keywords -> Patterns -> ML
pub struct NsfwCascade {
    stages: Vec<Box<dyn NsfwStage>>,
    has_ml: bool,
}

impl NsfwCascade {
    /// Create a new NSFW cascade with all default stages.
    pub fn new() -> Result<Self, Error> {
        let mut stages: Vec<Box<dyn NsfwStage>> = vec![
            Box::new(NsfwKeywordStage::new()),
            Box::new(NsfwPatternStage::new()),
        ];

        // Try to add ML stage, but don't fail if it's unavailable
        let has_ml = match NsfwMlStage::new() {
            Ok(ml_stage) => {
                stages.push(Box::new(ml_stage));
                true
            }
            Err(e) => {
                eprintln!("Warning: NSFW ML stage unavailable: {}", e);
                false
            }
        };

        Ok(Self { stages, has_ml })
    }

    /// Check if ML stage is available.
    pub fn has_ml_stage(&self) -> bool {
        self.has_ml
    }

    /// Classify content for NSFW.
    ///
    /// Returns a result from the first stage that provides one,
    /// or from the ML stage (which always returns a result).
    /// If ML stage is unavailable, returns a low-confidence "not nsfw" result.
    pub fn classify(&self, info: &ContentInfo) -> Result<NsfwResult, Error> {
        for stage in &self.stages {
            if let Some(result) = stage.classify(info)? {
                return Ok(result);
            }
        }

        // If we get here without ML, return low-confidence not-nsfw
        // (ML stage always returns a result when available)
        Ok(NsfwResult::new(false, 0.5, NsfwSource::Ml))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsfw_source_as_str() {
        assert_eq!(NsfwSource::Keywords.as_str(), "keywords");
        assert_eq!(NsfwSource::Patterns.as_str(), "patterns");
        assert_eq!(NsfwSource::Ml.as_str(), "ml");
    }

    #[test]
    fn test_nsfw_result_creation() {
        let result = NsfwResult::new(true, 0.95, NsfwSource::Keywords);
        assert!(result.nsfw);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.source, NsfwSource::Keywords);
    }

    #[test]
    fn test_nsfw_cascade_creation() {
        let cascade = NsfwCascade::new();
        assert!(cascade.is_ok());
    }

    #[test]
    fn test_nsfw_cascade_safe_content() {
        let cascade = NsfwCascade::new().unwrap();
        let info = ContentInfo {
            name: "The.Matrix.1999.1080p.BluRay".to_string(),
            files: vec![],
        };
        let result = cascade.classify(&info).unwrap();
        assert!(!result.nsfw);
    }

    #[test]
    fn test_nsfw_cascade_nsfw_keyword() {
        let cascade = NsfwCascade::new().unwrap();
        let info = ContentInfo {
            name: "Brazzers.XXX.Scene.1080p".to_string(),
            files: vec![],
        };
        let result = cascade.classify(&info).unwrap();
        assert!(result.nsfw);
        assert_eq!(result.source, NsfwSource::Keywords);
    }
}
