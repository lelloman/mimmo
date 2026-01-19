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
  "series_title": "string (required)",
  "episode_title": "string (optional)",
  "season_number": "number (optional) - HEURISTIC: extracted via regex (S01E01, 1x01)",
  "episode_number": "number (optional) - HEURISTIC: extracted via regex (S01E01, 1x01)",
  "year": "number (optional)"
}
```

**video/season:**
```json
{
  "series_title": "string (required)",
  "season_number": "number (optional) - HEURISTIC: extracted via regex or directory structure",
  "year": "number (optional)"
}
```

**video/series:**
```json
{
  "series_title": "string (required)"
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

---

## Entity Extraction Model Evaluation (2025-01-18)

### Models Tested

| Model | Params | Speed | Quality |
|-------|--------|-------|---------|
| **GLiNER multi-v2.1** | 209M | ~13/sec (75ms) | Good for audio, mixed for video |
| **NuExtract 4B** | 3.8B | ~0.5/sec (2000ms) | Better semantic understanding |
| **NuExtract-tiny** | 464M | ~0.9/sec (1100ms) | Poor - hallucinations, confusion |

### GLiNER Findings

**Audio extraction** (good):
- Artist/album/year extraction works well with `"Artist - Album (Year)"` format
- Examples: `Kathryn Williams - 2004 - Relations` → artist, year, album correctly extracted
- Struggles with ambiguous single names (e.g., "LA FLOA MALDITA" → extracted as album, but it's artist)

**Video extraction** (mixed):
- Title extraction often includes too much (e.g., whole torrent name)
- Season/episode parsing inconsistent (S05E16 sometimes in season field, episode number in another)
- Confuses names with metadata (e.g., "Autumn Falls" person name → season: "Autumn", episode: "Falls")
- Resolution/codec extraction works reasonably well

### NuExtract Findings

**4B version** (better quality, too slow):
- Better semantic understanding - correctly identified "Felina" as episode title (Breaking Bad finale)
- Understands context better than GLiNER
- JSON schema output is cleaner
- 20-25x slower than GLiNER - not practical for bulk labeling

**Tiny version** (not recommended):
- Frequent hallucinations (generated unrelated text about Billboard Hot 100)
- Confused field mappings worse than GLiNER
- Slower than GLiNER despite being smaller

### Performance Summary

For labeling 100k samples:
- **GLiNER**: ~2 hours
- **NuExtract 4B**: ~56 hours
- **NuExtract-tiny**: ~31 hours (with poor quality)

### Mimmo Classification Speed

| Method | Speed | Per Sample |
|--------|-------|------------|
| Mimmo binary (direct) | ~1470/sec | 0.68ms |
| Mimmo via subprocess | ~5.7/sec | 175ms |

Classification is essentially instant compared to extraction.

### Conclusions

1. **GLiNER is the best option** for bulk labeling due to speed
2. **Quality is acceptable** but not perfect - fine for training data generation
3. **Heuristic extraction** (Rust regex, no ML needed):
   - Technical: resolution, codec, bitrate, containers
   - Video: season_number, episode_number (S01E01 patterns)
4. **ML extraction** (needs NER/model):
   - Video: series_title, episode_title, movie title, year
   - Audio: artist, album, track_name, year

### Recommended Approach

**Option A: Distillation Pipeline**
1. Use GLiNER to label 100k+ samples
2. Train smaller token classifier (~50M params) on those labels
3. Embed in mimmo binary

**Option B: Hybrid Regex + Optional ML**
1. Implement regex extraction for technical metadata in Rust
2. Keep semantic extraction as optional Python/GLiNER dependency
3. Simpler, but no embedded extraction

**Option C: Use LLM for Higher Quality Labels**
1. Use local LLM (qwen2.5:3b) for labeling instead of GLiNER
2. Slower but potentially better quality training data
3. Then distill to smaller model
