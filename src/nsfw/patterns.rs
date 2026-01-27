//! Pattern-based NSFW detection stage.
//!
//! Uses regex patterns to detect NSFW content patterns like JAV codes,
//! studio naming conventions, and scene descriptors.

use regex::Regex;
use std::sync::OnceLock;

use crate::nsfw::{NsfwResult, NsfwSource, NsfwStage};
use crate::{ContentInfo, Error};

/// Stage that checks for NSFW-specific naming patterns.
pub struct NsfwPatternStage {
    _private: (),
}

impl NsfwPatternStage {
    /// Create a new pattern stage.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for NsfwPatternStage {
    fn default() -> Self {
        Self::new()
    }
}

impl NsfwStage for NsfwPatternStage {
    fn name(&self) -> &'static str {
        "nsfw_patterns"
    }

    fn classify(&self, info: &ContentInfo) -> Result<Option<NsfwResult>, Error> {
        // Check torrent name
        if matches_nsfw_pattern(&info.name) {
            return Ok(Some(NsfwResult::new(true, 0.90, NsfwSource::Patterns)));
        }

        // Check file names
        for file in &info.files {
            if matches_nsfw_pattern(&file.filename) {
                return Ok(Some(NsfwResult::new(true, 0.90, NsfwSource::Patterns)));
            }
        }

        Ok(None)
    }
}

/// Check if text matches any NSFW pattern.
fn matches_nsfw_pattern(text: &str) -> bool {
    // JAV codes: SNIS-123, ABP-456, MIDE-789, etc.
    static JAV_CODE: OnceLock<Regex> = OnceLock::new();
    let jav_code = JAV_CODE.get_or_init(|| {
        Regex::new(r"(?i)\b[A-Z]{2,6}-\d{3,5}\b").unwrap()
    });

    // Studio + date patterns: brazzers.24.01.15.performer, realitykings.2024.01.15
    static STUDIO_DATE: OnceLock<Regex> = OnceLock::new();
    let studio_date = STUDIO_DATE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(brazzers|bangbros|realitykings|naughtyamerica|mofos|
               fakehub|faketaxi|publicagent|vixen|blacked|tushy|deeper|
               digitalplayground|wicked|evil\s*angel|jules\s*jordan|
               girlsway|sweetsinner|new\s*sensations|elegantangel)
            \s*[\._-]?\s*
            (\d{2,4}[\._-]\d{2}[\._-]\d{2}|\d{4}[\._-]\d{2}[\._-]\d{2})
            "
        ).unwrap()
    });

    // Scene descriptors with names: "performer.and.performer.scene"
    static SCENE_PATTERN: OnceLock<Regex> = OnceLock::new();
    let scene_pattern = SCENE_PATTERN.get_or_init(|| {
        Regex::new(
            r"(?ix)
            # Multiple performers pattern
            [a-z]+\s*[\._]\s*and\s*[\._]\s*[a-z]+\s*[\._-]
            |
            # Explicit scene descriptors
            \b(big[\._-]?tits|big[\._-]?ass|big[\._-]?cock|
               hot[\._-]?mom|step[\._-]?mom|step[\._-]?sis|step[\._-]?bro|
               teen[\._-]?slut|college[\._-]?girl|horny[\._-]?wife)\b
            |
            # 18+ indicators in torrent names
            \b18\+|
            \b(adults[\._-]?only|over[\._-]?18)\b
            "
        ).unwrap()
    });

    // OnlyFans/Fansly creator patterns
    static CREATOR_PATTERN: OnceLock<Regex> = OnceLock::new();
    let creator_pattern = CREATOR_PATTERN.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(onlyfans|fansly|manyvids)\s*[\._-]?\s*[a-z]+
            |
            [a-z]+\s*[\._-]?\s*(siterip|megapack)\b
            "
        ).unwrap()
    });

    // FC2-PPV pattern (Japanese paid content)
    static FC2_PATTERN: OnceLock<Regex> = OnceLock::new();
    let fc2_pattern = FC2_PATTERN.get_or_init(|| {
        Regex::new(r"(?i)\bFC2[\._-]?PPV[\._-]?\d+\b").unwrap()
    });

    // Heydouga/Caribbean/1pondo patterns
    static JP_STUDIO_PATTERN: OnceLock<Regex> = OnceLock::new();
    let jp_studio_pattern = JP_STUDIO_PATTERN.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b(heydouga|caribbeancom|caribbean|1pondo|10musume|
               pacopacomama|heyzo|tokyo[\._-]?hot|mesubuta)\b
            \s*[\._-]?\s*
            (\d{4,}|\d{2,4}[\._-]\d{2,4})
            "
        ).unwrap()
    });

    // Check all patterns
    jav_code.is_match(text)
        || studio_date.is_match(text)
        || scene_pattern.is_match(text)
        || creator_pattern.is_match(text)
        || fc2_pattern.is_match(text)
        || jp_studio_pattern.is_match(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileInfo;

    fn classify_name(name: &str) -> Option<NsfwResult> {
        let stage = NsfwPatternStage::new();
        let info = ContentInfo {
            name: name.to_string(),
            files: vec![],
        };
        stage.classify(&info).unwrap()
    }

    fn classify_with_files(name: &str, filenames: &[&str]) -> Option<NsfwResult> {
        let stage = NsfwPatternStage::new();
        let files = filenames
            .iter()
            .map(|f| FileInfo {
                path: f.to_string(),
                filename: f.to_string(),
                size: 1000,
            })
            .collect();
        let info = ContentInfo {
            name: name.to_string(),
            files,
        };
        stage.classify(&info).unwrap()
    }

    // JAV code tests
    #[test]
    fn test_jav_code_snis() {
        let result = classify_name("SNIS-123.1080p");
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.nsfw);
        assert_eq!(r.source, NsfwSource::Patterns);
    }

    #[test]
    fn test_jav_code_abp() {
        let result = classify_name("ABP-456.Scene.720p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_jav_code_mide() {
        let result = classify_name("MIDE-789.Name.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_jav_code_long() {
        let result = classify_name("CAWD-12345.Scene");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    // Studio + date pattern tests
    #[test]
    fn test_studio_date_brazzers() {
        let result = classify_name("Brazzers.24.01.15.Performer.Name.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_studio_date_realitykings() {
        let result = classify_name("RealityKings.2024.01.15.Scene.Name");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_studio_date_vixen() {
        let result = classify_name("Vixen.24.03.22.Star.Name.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    // Scene pattern tests
    #[test]
    fn test_scene_descriptors() {
        let result = classify_name("Hot.Mom.Teaches.Lesson.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_adults_only() {
        let result = classify_name("Content.Adults.Only.Version");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_18_plus() {
        let result = classify_name("Movie.18+.Uncut.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    // FC2 pattern tests
    #[test]
    fn test_fc2_ppv() {
        let result = classify_name("FC2-PPV-1234567");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_fc2_ppv_alt_format() {
        let result = classify_name("FC2PPV1234567.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    // Japanese studio pattern tests
    #[test]
    fn test_heydouga() {
        let result = classify_name("Heydouga.4030-2234");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_caribbeancom() {
        let result = classify_name("Caribbeancom.123456");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_1pondo() {
        let result = classify_name("1pondo.123456_001");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    // OnlyFans/creator pattern tests
    #[test]
    fn test_onlyfans_creator() {
        let result = classify_name("OnlyFans.CreatorName.Pack");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_siterip_pattern() {
        let result = classify_name("CreatorName.Siterip.2024");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_megapack_pattern() {
        let result = classify_name("ModelName.Megapack.Complete");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    // Safe content tests
    #[test]
    fn test_safe_movie() {
        let result = classify_name("The.Matrix.1999.1080p.BluRay");
        assert!(result.is_none());
    }

    #[test]
    fn test_safe_tv_show() {
        let result = classify_name("Breaking.Bad.S01E01.720p");
        assert!(result.is_none());
    }

    #[test]
    fn test_safe_software() {
        let result = classify_name("Adobe.Photoshop.2024.v25.1");
        assert!(result.is_none());
    }

    // File name detection tests
    #[test]
    fn test_file_name_jav() {
        let result = classify_with_files("Torrent.Name", &["SNIS-123.mp4"]);
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_file_names_safe() {
        let result = classify_with_files("Movie.2024", &["movie.mkv", "subs.srt"]);
        assert!(result.is_none());
    }
}
