//! Core types for the cascade classification system.

use std::fmt;

/// The medium type (content category).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Medium {
    Video,
    Audio,
    Book,
    Software,
    Other,
}

impl Medium {
    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Medium::Video => "video",
            Medium::Audio => "audio",
            Medium::Book => "book",
            Medium::Software => "software",
            Medium::Other => "other",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "video" => Some(Medium::Video),
            "audio" => Some(Medium::Audio),
            "book" => Some(Medium::Book),
            "software" => Some(Medium::Software),
            "other" => Some(Medium::Other),
            _ => None,
        }
    }
}

impl fmt::Display for Medium {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Confidence level of a classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Confidence {
    /// Uncertain / fallback classification.
    Low,
    /// Probable match based on heuristics.
    Medium,
    /// Strong deterministic match.
    High,
}

impl Confidence {
    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Confidence::Low => "low",
            Confidence::Medium => "medium",
            Confidence::High => "high",
        }
    }

    /// Convert to a numeric score (0.0 to 1.0).
    pub fn as_score(&self) -> f32 {
        match self {
            Confidence::Low => 0.3,
            Confidence::Medium => 0.7,
            Confidence::High => 0.95,
        }
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Result from a classification stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// The classified medium type.
    pub medium: Medium,
    /// Confidence level of this classification.
    pub confidence: Confidence,
    /// Name of the stage that produced this result.
    pub source: &'static str,
}

impl StageResult {
    /// Create a new stage result.
    pub fn new(medium: Medium, confidence: Confidence, source: &'static str) -> Self {
        Self {
            medium,
            confidence,
            source,
        }
    }

    /// Create a high-confidence result.
    pub fn high(medium: Medium, source: &'static str) -> Self {
        Self::new(medium, Confidence::High, source)
    }

    /// Create a medium-confidence result.
    pub fn medium(medium: Medium, source: &'static str) -> Self {
        Self::new(medium, Confidence::Medium, source)
    }

    /// Create a low-confidence result.
    pub fn low(medium: Medium, source: &'static str) -> Self {
        Self::new(medium, Confidence::Low, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medium_as_str() {
        assert_eq!(Medium::Video.as_str(), "video");
        assert_eq!(Medium::Audio.as_str(), "audio");
        assert_eq!(Medium::Book.as_str(), "book");
        assert_eq!(Medium::Software.as_str(), "software");
        assert_eq!(Medium::Other.as_str(), "other");
    }

    #[test]
    fn test_medium_from_str() {
        assert_eq!(Medium::from_str("video"), Some(Medium::Video));
        assert_eq!(Medium::from_str("VIDEO"), Some(Medium::Video));
        assert_eq!(Medium::from_str("unknown"), None);
    }

    #[test]
    fn test_confidence_ordering() {
        assert!(Confidence::Low < Confidence::Medium);
        assert!(Confidence::Medium < Confidence::High);
    }

    #[test]
    fn test_confidence_scores() {
        assert!(Confidence::Low.as_score() < Confidence::Medium.as_score());
        assert!(Confidence::Medium.as_score() < Confidence::High.as_score());
    }
}
