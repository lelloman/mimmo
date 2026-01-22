//! Format extraction from torrent names.
//!
//! This module parses torrent names to extract format information like
//! resolution, codecs, quality tier, and episode numbers.

use once_cell::sync::Lazy;
use regex::Regex;

/// Video resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resolution {
    R480p,
    R720p,
    R1080p,
    R2160p,
}

/// Video codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
    AV1,
    VP9,
}

/// Audio codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    AAC,
    AC3,
    DTS,
    DtsHd,
    TrueHD,
    Atmos,
    FLAC,
}

/// Quality tier ranking (ordered from lowest to highest quality).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityTier {
    Cam,
    TeleSync,
    TeleCine,
    Screener,
    DVDRip,
    HDTV,
    WebRip,
    WebDL,
    BluRay,
    Remux,
}

impl QualityTier {
    /// Get the numeric rank (0-9) for scoring.
    pub fn rank(&self) -> u8 {
        match self {
            QualityTier::Cam => 0,
            QualityTier::TeleSync => 1,
            QualityTier::TeleCine => 2,
            QualityTier::Screener => 3,
            QualityTier::DVDRip => 4,
            QualityTier::HDTV => 5,
            QualityTier::WebRip => 6,
            QualityTier::WebDL => 7,
            QualityTier::BluRay => 8,
            QualityTier::Remux => 9,
        }
    }
}

/// Container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    MKV,
    MP4,
    AVI,
}

/// Extracted format information from a torrent name.
#[derive(Debug, Clone, Default)]
pub struct ParsedFormat {
    pub resolution: Option<Resolution>,
    pub video_codec: Option<VideoCodec>,
    pub audio_codec: Option<AudioCodec>,
    pub quality_tier: Option<QualityTier>,
    pub container: Option<Container>,
    pub season: Option<u16>,
    pub episode: Option<u16>,
}

// Regex patterns
static RESOLUTION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(2160p|4K|UHD|1080p|720p|480p)\b").unwrap()
});

static VIDEO_CODEC_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(x\.?264|x\.?265|H\.?264|H\.?265|HEVC|AVC|AV1|VP9)\b").unwrap()
});

static AUDIO_CODEC_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(AAC|AC3|DD5\.?1|DTS-HD(?:[\.-]?MA)?|DTS|TrueHD|Atmos|FLAC)\b").unwrap()
});

static QUALITY_TIER_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        \b(
            CAM(?:RIP)?|
            TS|TELESYNC|HDTS|
            TC|TELECINE|
            SCR|SCREENER|DVDSCR|
            DVD(?:RIP)?|
            HDTV(?:RIP)?|
            WEB-?DL|
            WEB-?RIP|
            BLU-?RAY|BD(?:RIP)?|BRRIP|
            REMUX
        )\b",
    )
    .unwrap()
});

static CONTAINER_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\.?(mkv|mp4|avi)\b").unwrap()
});

static SEASON_EPISODE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bS(\d{1,2})E(\d{1,3})\b").unwrap()
});

// Season-only pattern: S followed by digits, then word boundary or non-E character
// We'll check for "SnnE" separately in code since regex doesn't support look-ahead
static SEASON_ONLY_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bS(\d{1,2})\b").unwrap()
});

/// Parse format information from a torrent name.
pub fn parse_format(name: &str) -> ParsedFormat {
    let mut format = ParsedFormat::default();

    // Parse resolution
    if let Some(caps) = RESOLUTION_PATTERN.captures(name) {
        let res = caps.get(1).unwrap().as_str().to_uppercase();
        format.resolution = match res.as_str() {
            "480P" => Some(Resolution::R480p),
            "720P" => Some(Resolution::R720p),
            "1080P" => Some(Resolution::R1080p),
            "2160P" | "4K" | "UHD" => Some(Resolution::R2160p),
            _ => None,
        };
    }

    // Parse video codec
    if let Some(caps) = VIDEO_CODEC_PATTERN.captures(name) {
        let codec = caps.get(1).unwrap().as_str().to_uppercase();
        let codec = codec.replace('.', "");
        format.video_codec = match codec.as_str() {
            "X264" | "H264" | "AVC" => Some(VideoCodec::H264),
            "X265" | "H265" | "HEVC" => Some(VideoCodec::H265),
            "AV1" => Some(VideoCodec::AV1),
            "VP9" => Some(VideoCodec::VP9),
            _ => None,
        };
    }

    // Parse audio codec
    if let Some(caps) = AUDIO_CODEC_PATTERN.captures(name) {
        let codec = caps.get(1).unwrap().as_str().to_uppercase();
        format.audio_codec = if codec.starts_with("DTS-HD") || codec.starts_with("DTS.HD") {
            Some(AudioCodec::DtsHd)
        } else {
            match codec.as_str() {
                "AAC" => Some(AudioCodec::AAC),
                "AC3" | "DD51" | "DD5.1" => Some(AudioCodec::AC3),
                "DTS" => Some(AudioCodec::DTS),
                "TRUEHD" => Some(AudioCodec::TrueHD),
                "ATMOS" => Some(AudioCodec::Atmos),
                "FLAC" => Some(AudioCodec::FLAC),
                _ => None,
            }
        };
    }

    // Parse quality tier
    if let Some(caps) = QUALITY_TIER_PATTERN.captures(name) {
        let tier = caps.get(1).unwrap().as_str().to_uppercase();
        let tier = tier.replace('-', "");
        format.quality_tier = match tier.as_str() {
            "CAM" | "CAMRIP" => Some(QualityTier::Cam),
            "TS" | "TELESYNC" | "HDTS" => Some(QualityTier::TeleSync),
            "TC" | "TELECINE" => Some(QualityTier::TeleCine),
            "SCR" | "SCREENER" | "DVDSCR" => Some(QualityTier::Screener),
            "DVD" | "DVDRIP" => Some(QualityTier::DVDRip),
            "HDTV" | "HDTVRIP" => Some(QualityTier::HDTV),
            "WEBDL" => Some(QualityTier::WebDL),
            "WEBRIP" => Some(QualityTier::WebRip),
            "BLURAY" | "BD" | "BDRIP" | "BRRIP" => Some(QualityTier::BluRay),
            "REMUX" => Some(QualityTier::Remux),
            _ => None,
        };
    }

    // Parse container
    if let Some(caps) = CONTAINER_PATTERN.captures(name) {
        let container = caps.get(1).unwrap().as_str().to_uppercase();
        format.container = match container.as_str() {
            "MKV" => Some(Container::MKV),
            "MP4" => Some(Container::MP4),
            "AVI" => Some(Container::AVI),
            _ => None,
        };
    }

    // Parse season/episode - try full S##E## pattern first
    if let Some(caps) = SEASON_EPISODE_PATTERN.captures(name) {
        if let Ok(season) = caps.get(1).unwrap().as_str().parse::<u16>() {
            format.season = Some(season);
        }
        if let Ok(episode) = caps.get(2).unwrap().as_str().parse::<u16>() {
            format.episode = Some(episode);
        }
    } else if let Some(caps) = SEASON_ONLY_PATTERN.captures(name) {
        // Only use season-only pattern if there's no S##E## match
        // Check that this isn't actually a S##E## pattern by looking at what follows
        let match_end = caps.get(0).unwrap().end();
        let remainder = &name[match_end..];
        // If followed by 'E' and digits, skip (it's a partial match of S##E##)
        if !remainder.starts_with('E')
            && !remainder.starts_with('e')
        {
            if let Ok(season) = caps.get(1).unwrap().as_str().parse::<u16>() {
                format.season = Some(season);
            }
        }
    }

    format
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_movie_bluray() {
        let format = parse_format("Inception.2010.1080p.BluRay.x264-GROUP");
        assert_eq!(format.resolution, Some(Resolution::R1080p));
        assert_eq!(format.quality_tier, Some(QualityTier::BluRay));
        assert_eq!(format.video_codec, Some(VideoCodec::H264));
        assert!(format.season.is_none());
        assert!(format.episode.is_none());
    }

    #[test]
    fn test_parse_tv_episode() {
        let format = parse_format("Breaking.Bad.S05E16.720p.WEB-DL.DD5.1.H.264");
        assert_eq!(format.resolution, Some(Resolution::R720p));
        assert_eq!(format.quality_tier, Some(QualityTier::WebDL));
        assert_eq!(format.audio_codec, Some(AudioCodec::AC3));
        assert_eq!(format.video_codec, Some(VideoCodec::H264));
        assert_eq!(format.season, Some(5));
        assert_eq!(format.episode, Some(16));
    }

    #[test]
    fn test_parse_4k_hevc() {
        let format = parse_format("Movie.2024.2160p.UHD.BluRay.HEVC.Atmos");
        assert_eq!(format.resolution, Some(Resolution::R2160p));
        assert_eq!(format.quality_tier, Some(QualityTier::BluRay));
        assert_eq!(format.video_codec, Some(VideoCodec::H265));
        assert_eq!(format.audio_codec, Some(AudioCodec::Atmos));
    }

    #[test]
    fn test_parse_remux() {
        let format = parse_format("Movie.2020.1080p.Remux.AVC.TrueHD.7.1");
        assert_eq!(format.resolution, Some(Resolution::R1080p));
        assert_eq!(format.quality_tier, Some(QualityTier::Remux));
        assert_eq!(format.video_codec, Some(VideoCodec::H264));
        assert_eq!(format.audio_codec, Some(AudioCodec::TrueHD));
    }

    #[test]
    fn test_parse_webrip() {
        let format = parse_format("Show.S01E01.WEBRip.x265.AAC");
        assert_eq!(format.quality_tier, Some(QualityTier::WebRip));
        assert_eq!(format.video_codec, Some(VideoCodec::H265));
        assert_eq!(format.audio_codec, Some(AudioCodec::AAC));
        assert_eq!(format.season, Some(1));
        assert_eq!(format.episode, Some(1));
    }

    #[test]
    fn test_parse_cam() {
        let format = parse_format("New.Movie.2024.CAM.x264");
        assert_eq!(format.quality_tier, Some(QualityTier::Cam));
        assert_eq!(format.video_codec, Some(VideoCodec::H264));
    }

    #[test]
    fn test_parse_hdtv() {
        let format = parse_format("Show.S02E05.HDTV.720p.AC3");
        assert_eq!(format.quality_tier, Some(QualityTier::HDTV));
        assert_eq!(format.resolution, Some(Resolution::R720p));
        assert_eq!(format.audio_codec, Some(AudioCodec::AC3));
        assert_eq!(format.season, Some(2));
        assert_eq!(format.episode, Some(5));
    }

    #[test]
    fn test_parse_dts_hd() {
        let format = parse_format("Movie.2020.1080p.BluRay.DTS-HD.MA");
        assert_eq!(format.audio_codec, Some(AudioCodec::DtsHd));
    }

    #[test]
    fn test_parse_container() {
        let format = parse_format("Movie.2020.1080p.BluRay.x264.mkv");
        assert_eq!(format.container, Some(Container::MKV));

        let format = parse_format("Movie.2020.720p.WebRip.mp4");
        assert_eq!(format.container, Some(Container::MP4));
    }

    #[test]
    fn test_parse_av1() {
        let format = parse_format("Movie.2024.1080p.WEB-DL.AV1.AAC");
        assert_eq!(format.video_codec, Some(VideoCodec::AV1));
    }

    #[test]
    fn test_parse_season_only() {
        let format = parse_format("Show.S03.Complete.720p.BluRay");
        assert_eq!(format.season, Some(3));
        assert!(format.episode.is_none());
    }

    #[test]
    fn test_parse_4k_alias() {
        let format = parse_format("Movie.2024.4K.HDR.WEB-DL");
        assert_eq!(format.resolution, Some(Resolution::R2160p));
    }

    #[test]
    fn test_parse_no_match() {
        let format = parse_format("random text without format info");
        assert!(format.resolution.is_none());
        assert!(format.video_codec.is_none());
        assert!(format.audio_codec.is_none());
        assert!(format.quality_tier.is_none());
        assert!(format.season.is_none());
        assert!(format.episode.is_none());
    }

    #[test]
    fn test_quality_tier_ranking() {
        assert!(QualityTier::Cam < QualityTier::TeleSync);
        assert!(QualityTier::TeleSync < QualityTier::DVDRip);
        assert!(QualityTier::DVDRip < QualityTier::HDTV);
        assert!(QualityTier::HDTV < QualityTier::WebRip);
        assert!(QualityTier::WebRip < QualityTier::WebDL);
        assert!(QualityTier::WebDL < QualityTier::BluRay);
        assert!(QualityTier::BluRay < QualityTier::Remux);
    }

    #[test]
    fn test_quality_tier_rank() {
        assert_eq!(QualityTier::Cam.rank(), 0);
        assert_eq!(QualityTier::Remux.rank(), 9);
        assert_eq!(QualityTier::BluRay.rank(), 8);
    }
}
