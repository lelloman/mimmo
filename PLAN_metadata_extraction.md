# Metadata Extraction Plan

## Overview

Extract structured metadata from torrent content based on medium type and subcategory.

## Metadata Schemas

### Video

**video/movie, video/episode:**
```json
{
  "title": "string (required)",
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

### Technical Metadata (all media types)

```json
{
  "containers": "string[] (required)",
  "codec": "string (optional)",
  "bitrate": "string (optional)"
}
```

## Implementation Approach

### Technical Metadata (Heuristics)

- **Containers**: Extract from file extensions (mkv, mp4, avi, flac, mp3, etc.)
- **Codec**: Parse from filename patterns (x264, x265, HEVC, AV1, FLAC, AAC, etc.)
- **Bitrate**: Parse from filename patterns (320, 192, 24bit, 16bit, etc.)

### Content Metadata (ML/LLM)

TBD - Options:
1. Fine-tune a model for structured extraction
2. Use LLM with JSON schema prompting for labeling, then train smaller model
3. Rule-based extraction with regex fallbacks

## Open Questions

- [ ] How to handle content metadata extraction? (ML approach TBD)
- [ ] Should software and book types have metadata schemas?
- [ ] Video resolution as technical metadata? (1080p, 4K, etc.)
