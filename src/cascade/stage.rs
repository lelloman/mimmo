//! Stage trait and related utilities.

use crate::{ContentInfo, Error};

use super::StageResult;

/// A single stage in the classification cascade.
///
/// Each stage examines the content and either:
/// - Returns `Some(result)` if it can confidently classify
/// - Returns `None` to pass to the next stage
///
/// # Implementation Notes
///
/// - Stages should be fast and only return `Some` when confident
/// - Return `None` liberally - it's better to let the next stage try
/// - Use the `source` field in `StageResult` to identify your stage
pub trait Stage: Send + Sync {
    /// The name of this stage (for debugging/logging).
    fn name(&self) -> &'static str;

    /// Try to classify the content.
    ///
    /// Returns:
    /// - `Ok(Some(result))` if classification is confident
    /// - `Ok(None)` to pass to the next stage
    /// - `Err(e)` on error (will propagate up)
    fn classify(&self, info: &ContentInfo) -> Result<Option<StageResult>, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::Medium;

    struct TestStage {
        result: Option<StageResult>,
    }

    impl Stage for TestStage {
        fn name(&self) -> &'static str {
            "test"
        }

        fn classify(&self, _info: &ContentInfo) -> Result<Option<StageResult>, Error> {
            Ok(self.result.clone())
        }
    }

    #[test]
    fn test_stage_trait() {
        let stage = TestStage {
            result: Some(StageResult::high(Medium::Video, "test")),
        };
        assert_eq!(stage.name(), "test");
    }
}
