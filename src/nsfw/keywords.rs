//! Keyword-based NSFW detection stage.
//!
//! Uses a HashSet lookup for explicit terms. O(1) lookup, ~0.1ms per classification.

use std::collections::HashSet;

use crate::nsfw::{NsfwResult, NsfwSource, NsfwStage};
use crate::{ContentInfo, Error};

/// Keywords that strongly indicate NSFW content.
const NSFW_KEYWORDS: &[&str] = &[
    // Explicit content indicators
    "xxx",
    "porn",
    "porno",
    "hentai",
    "erotic",
    "erotica",
    "hardcore",
    "softcore",
    // Adult studios
    "brazzers",
    "bangbros",
    "realitykings",
    "naughtyamerica",
    "mofos",
    "fakehub",
    "faketaxi",
    "publicagent",
    "vixen",
    "blacked",
    "tushy",
    "deeper",
    "pornhub",
    "xvideos",
    "xhamster",
    "youporn",
    "redtube",
    // Japanese adult video
    "jav",
    "javhd",
    "caribbeancom",
    "tokyo-hot",
    "heyzo",
    "1pondo",
    "pacopacomama",
    "10musume",
    "muramura",
    // Other explicit terms
    "milf",
    "pawg",
    "bdsm",
    "femdom",
    "gangbang",
    "creampie",
    "blowjob",
    "handjob",
    "footjob",
    "bukkake",
    "facial",
    "cumshot",
    "doggystyle",
    "threesome",
    "foursome",
    "orgy",
    "anal",
    "dp",
    "lesbian",
    "lesbians",
    "shemale",
    "transsexual",
    "interracial",
    // Anime/hentai terms
    "uncensored",
    "ahegao",
    "ecchi",
    "oppai",
    // OnlyFans and similar
    "onlyfans",
    "fansly",
    "manyvids",
    "siterip",
];

/// Stage that checks for explicit NSFW keywords.
pub struct NsfwKeywordStage {
    keywords: HashSet<&'static str>,
}

impl NsfwKeywordStage {
    /// Create a new keyword stage.
    pub fn new() -> Self {
        let keywords: HashSet<&'static str> = NSFW_KEYWORDS.iter().copied().collect();
        Self { keywords }
    }

    /// Check if text contains any NSFW keywords.
    fn contains_keyword(&self, text: &str) -> bool {
        let lower = text.to_lowercase();

        // Check each word in the text
        for word in lower.split(|c: char| !c.is_alphanumeric()) {
            if self.keywords.contains(word) {
                return true;
            }
        }

        false
    }
}

impl Default for NsfwKeywordStage {
    fn default() -> Self {
        Self::new()
    }
}

impl NsfwStage for NsfwKeywordStage {
    fn name(&self) -> &'static str {
        "nsfw_keywords"
    }

    fn classify(&self, info: &ContentInfo) -> Result<Option<NsfwResult>, Error> {
        // Check torrent name
        if self.contains_keyword(&info.name) {
            return Ok(Some(NsfwResult::new(true, 0.95, NsfwSource::Keywords)));
        }

        // Check file names
        for file in &info.files {
            if self.contains_keyword(&file.filename) {
                return Ok(Some(NsfwResult::new(true, 0.95, NsfwSource::Keywords)));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileInfo;

    fn classify_name(name: &str) -> Option<NsfwResult> {
        let stage = NsfwKeywordStage::new();
        let info = ContentInfo {
            name: name.to_string(),
            files: vec![],
        };
        stage.classify(&info).unwrap()
    }

    fn classify_with_files(name: &str, filenames: &[&str]) -> Option<NsfwResult> {
        let stage = NsfwKeywordStage::new();
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

    #[test]
    fn test_explicit_xxx() {
        let result = classify_name("Some.XXX.Content.1080p");
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.nsfw);
        assert_eq!(r.source, NsfwSource::Keywords);
    }

    #[test]
    fn test_studio_brazzers() {
        let result = classify_name("Brazzers.Scene.Name.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_studio_bangbros() {
        let result = classify_name("BangBros.Scene.720p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_porn_keyword() {
        let result = classify_name("Some.Porn.Movie.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_hentai_keyword() {
        let result = classify_name("Anime.Hentai.Episode.01");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_jav_keyword() {
        let result = classify_name("JAV.Collection.2024");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_safe_content() {
        let result = classify_name("The.Matrix.1999.1080p.BluRay");
        assert!(result.is_none());
    }

    #[test]
    fn test_safe_content_with_similar_words() {
        // "anal" should not match "analysis"
        let result = classify_name("Data.Analysis.Course.2024");
        assert!(result.is_none());
    }

    #[test]
    fn test_file_names_nsfw() {
        let result = classify_with_files("Torrent.Name", &["scene1.xxx.mp4", "scene2.mp4"]);
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_file_names_safe() {
        let result = classify_with_files("Torrent.Name", &["movie.mkv", "subs.srt"]);
        assert!(result.is_none());
    }

    #[test]
    fn test_case_insensitive() {
        let result = classify_name("BRAZZERS.Scene.1080p");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }

    #[test]
    fn test_onlyfans_siterip() {
        let result = classify_name("OnlyFans.Creator.Siterip.2024");
        assert!(result.is_some());
        assert!(result.unwrap().nsfw);
    }
}
