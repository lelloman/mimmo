# Metadata Extraction Plan

## Overview

Extract structured metadata from torrent content based on medium type and subcategory.

## Metadata Schemas

### Video

**video/movie:**
```json
{
  "title": "string (required)",
  "year": "number (optional)"
}
```

**video/episode:**
```json
{
  "title": "string (required)",
  "series_title": "string (optional)",
  "season_number": "number (optional)",
  "episode_number": "number (optional)",
  "year": "number (optional)"
}
```

**video/season:**
```json
{
  "show_name": "string (required)",
  "season_number": "number (required)",
  "year": "number (optional)"
}
```

**video/series:**
```json
{
  "show_name": "string (required)",
  "season_count": "number (required)"
}
```

### Audio

**audio/track:**
```json
{
  "track_name": "string (required)",
  "artist": "string (optional)",
  "album": "string (optional)",
  "year": "number (optional)"
}
```

**audio/album:**
```json
{
  "album_name": "string (required)",
  "artist": "string (optional)",
  "year": "number (optional)",
  "track_count": "number (required)"
}
```

**audio/collection:**
```json
{
  "artists": "string[] (required)",
  "collection_name": "string (optional)"
}
```

### Technical Metadata (all audio/video types)

```json
{
  "containers": "string[] (required)",
  "codec": "string (optional)",
  "bitrate": "string (optional)",
  "resolution": "string (optional)"
}
```

## Implementation Approach

### Technical Metadata (Heuristics - Rust)

Can be extracted from filenames using pattern matching:

- **Containers**: File extensions (mkv, mp4, avi, flac, mp3, etc.)
- **Codec**: Filename patterns (x264, x265, HEVC, AV1, FLAC, AAC, etc.)
- **Bitrate**: Filename patterns (320, 192, 24bit, 16bit, V0, etc.)
- **Resolution**: Filename patterns (1080p, 720p, 4K, 2160p, etc.)

### Content Metadata (ML/LLM)

Options:
1. **LLM labeling → train smaller model**: Use LLM with JSON schema prompting to label dataset, then fine-tune a small model for extraction
2. **Direct LLM inference**: Use small LLM (Qwen2.5-0.5B) embedded in binary for extraction
3. **Hybrid**: Regex extraction for common patterns, LLM for ambiguous cases

## Implementation Order

1. [ ] Technical metadata extraction (heuristics in Rust)
2. [ ] Content metadata labeling pipeline (LLM)
3. [ ] Content metadata model training or integration

## Open Questions

- [ ] Should software and book types have metadata schemas?
- [ ] For audio/collection, how to handle Various Artists compilations?
- [ ] video/episode: Is series context always derivable from filename, or should we just extract title?
