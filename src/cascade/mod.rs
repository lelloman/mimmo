//! Cascade classification system.
//!
//! The cascade classifier runs a series of stages in order, returning
//! as soon as any stage produces a confident classification.
//!
//! # Example
//!
//! ```no_run
//! use mimmo::cascade::{Cascade, Medium};
//! use mimmo::ContentInfo;
//!
//! // Create cascade with default stages
//! let cascade = Cascade::default_with_ml().unwrap();
//!
//! let info = ContentInfo {
//!     name: "Movie.2024.1080p.BluRay.x264".to_string(),
//!     files: vec![],
//! };
//!
//! let result = cascade.classify(&info).unwrap();
//! assert_eq!(result.medium, Medium::Video);
//! ```

mod stage;
pub mod stages;
mod types;

#[cfg(test)]
mod samples_test;

pub use stage::Stage;
pub use stages::{ExtensionStage, MlStage, PatternStage};
pub use types::{Confidence, Medium, StageResult};

use crate::{ContentInfo, Error};

/// A cascade classifier that runs stages in order until one succeeds.
pub struct Cascade {
    stages: Vec<Box<dyn Stage>>,
}

impl Cascade {
    /// Create an empty cascade (no stages).
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Create a cascade with the given stages.
    pub fn with_stages(stages: Vec<Box<dyn Stage>>) -> Self {
        Self { stages }
    }

    /// Add a stage to the end of the cascade.
    pub fn add_stage<S: Stage + 'static>(&mut self, stage: S) {
        self.stages.push(Box::new(stage));
    }

    /// Classify content by running stages in order.
    ///
    /// Returns the first confident result, or falls back to `Other` with `Low` confidence.
    pub fn classify(&self, info: &ContentInfo) -> Result<StageResult, Error> {
        for stage in &self.stages {
            if let Some(result) = stage.classify(info)? {
                return Ok(result);
            }
        }

        // Final fallback
        Ok(StageResult {
            medium: Medium::Other,
            confidence: Confidence::Low,
            source: "fallback",
        })
    }

    /// Get the number of stages in the cascade.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Default for Cascade {
    fn default() -> Self {
        Self::new()
    }
}

impl Cascade {
    /// Create a cascade with the default heuristic stages (no ML).
    ///
    /// Stages: ExtensionStage → PatternStage → fallback
    pub fn default_heuristics() -> Self {
        let mut cascade = Self::new();
        cascade.add_stage(ExtensionStage::new());
        cascade.add_stage(PatternStage::new());
        cascade
    }

    /// Create a cascade with all default stages including ML fallback.
    ///
    /// Stages: ExtensionStage → PatternStage → MlStage → fallback
    pub fn default_with_ml() -> Result<Self, Error> {
        let mut cascade = Self::default_heuristics();
        cascade.add_stage(MlStage::new()?);
        Ok(cascade)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysVideo;

    impl Stage for AlwaysVideo {
        fn name(&self) -> &'static str {
            "always_video"
        }

        fn classify(&self, _info: &ContentInfo) -> Result<Option<StageResult>, Error> {
            Ok(Some(StageResult {
                medium: Medium::Video,
                confidence: Confidence::High,
                source: "always_video",
            }))
        }
    }

    struct NeverMatches;

    impl Stage for NeverMatches {
        fn name(&self) -> &'static str {
            "never_matches"
        }

        fn classify(&self, _info: &ContentInfo) -> Result<Option<StageResult>, Error> {
            Ok(None)
        }
    }

    #[test]
    fn test_empty_cascade_returns_fallback() {
        let cascade = Cascade::new();
        let info = ContentInfo {
            name: "test".to_string(),
            files: vec![],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Other);
        assert_eq!(result.confidence, Confidence::Low);
    }

    #[test]
    fn test_first_match_wins() {
        let mut cascade = Cascade::new();
        cascade.add_stage(AlwaysVideo);
        cascade.add_stage(NeverMatches);

        let info = ContentInfo {
            name: "test".to_string(),
            files: vec![],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Video);
        assert_eq!(result.confidence, Confidence::High);
    }

    #[test]
    fn test_skips_non_matching_stages() {
        let mut cascade = Cascade::new();
        cascade.add_stage(NeverMatches);
        cascade.add_stage(AlwaysVideo);

        let info = ContentInfo {
            name: "test".to_string(),
            files: vec![],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Video);
    }

    // Integration tests with real stages
    #[test]
    fn test_default_heuristics_cascade() {
        let cascade = Cascade::default_heuristics();
        assert_eq!(cascade.stage_count(), 2);
    }

    #[test]
    fn test_default_ml_cascade() {
        let cascade = Cascade::default_with_ml().unwrap();
        assert_eq!(cascade.stage_count(), 3);
    }

    #[test]
    fn test_cascade_video_by_extension() {
        let cascade = Cascade::default_heuristics();
        let info = ContentInfo {
            name: "Something".to_string(),
            files: vec![crate::FileInfo {
                path: "movie.mkv".to_string(),
                filename: "movie.mkv".to_string(),
                size: 1_000_000_000,
            }],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Video);
        assert_eq!(result.confidence, Confidence::High);
        assert_eq!(result.source, "extensions");
    }

    #[test]
    fn test_cascade_video_by_pattern() {
        let cascade = Cascade::default_heuristics();
        let info = ContentInfo {
            name: "Movie.2024.1080p.BluRay.x264-GROUP".to_string(),
            files: vec![],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Video);
        assert_eq!(result.source, "patterns");
    }

    #[test]
    fn test_cascade_audio_by_pattern() {
        let cascade = Cascade::default_heuristics();
        let info = ContentInfo {
            name: "Artist - Album (2024) [FLAC]".to_string(),
            files: vec![],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Audio);
    }

    #[test]
    fn test_cascade_software_by_pattern() {
        let cascade = Cascade::default_heuristics();
        let info = ContentInfo {
            name: "Adobe Photoshop CC 2024 + Crack".to_string(),
            files: vec![],
        };

        let result = cascade.classify(&info).unwrap();
        assert_eq!(result.medium, Medium::Software);
    }

    #[test]
    fn test_cascade_ml_fallback() {
        let cascade = Cascade::default_with_ml().unwrap();
        // A name that heuristics won't match but ML might
        let info = ContentInfo {
            name: "random name".to_string(),
            files: vec![crate::FileInfo {
                path: "video.mkv".to_string(),
                filename: "video.mkv".to_string(),
                size: 1_000_000_000,
            }],
        };

        let result = cascade.classify(&info).unwrap();
        // Should be classified by extensions stage first
        assert_eq!(result.medium, Medium::Video);
        assert_eq!(result.source, "extensions");
    }
}
