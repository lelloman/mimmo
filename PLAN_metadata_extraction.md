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
  "episode_number": "number (optional) - HEURISTIC: extracted via regex (S01E01, 1x01)"
}
```

**video/season:**
```json
{
  "series_title": "string (required)",
  "season_number": "number (optional) - HEURISTIC: extracted via regex or directory structure"
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
  "year": "number (optional)"
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
  "resolution": "string (optional)",
  "audio_languages": "string[] (optional)",
  "subtitle_languages": "string[] (optional)"
}
```

## Implementation Approach

### Cascade Extraction Pipeline

To avoid false positives (e.g., matching "English" in a title as a language), we use a cascade:

```
Input: "Pink Floyd - Dark Side of the Moon (1973) [FLAC 24bit 1080p] MULTI"
                              │
                              ▼
                   ┌─────────────────────┐
                   │  1. ML Extraction   │
                   │  (title, artist,    │
                   │   album, year)      │
                   └─────────────────────┘
                              │
          Extracted: "Pink Floyd", "Dark Side of the Moon", "1973"
                              │
                              ▼
                   ┌─────────────────────┐
                   │  2. Remove ML spans │
                   │  from input         │
                   └─────────────────────┘
                              │
          Remaining: " -  () [FLAC 24bit 1080p] MULTI"
                              │
                              ▼
                   ┌─────────────────────┐
                   │  3. Regex Extraction│
                   │  (technical metadata│
                   │   on remainder)     │
                   └─────────────────────┘
                              │
          Extracted: codec=FLAC, bitrate=24bit, resolution=1080p, audio_languages=[MULTI]
```

### Step 1: Content Metadata (ML)

Extract semantic content first:
- **Video**: title, series_title, year
- **Audio**: artist, album, album_name, track_name, year, collection_name, artists[]

### Step 2: Remove ML Spans

Remove the extracted content spans from the input text, leaving only the technical "cruft".

### Step 3: Technical Metadata (Regex on remainder)

Pattern matching on the remaining text:

**From filenames/structure:**
- **Containers**: File extensions (mkv, mp4, avi, flac, mp3, etc.)
- **season_number**: Regex patterns (S01, Season 1, etc.)
- **episode_number**: Regex patterns (E01, 1x01, etc.)

**From remaining text after ML extraction:**
- **Codec**: Patterns (x264, x265, HEVC, AV1, FLAC, AAC, etc.)
- **Bitrate**: Patterns (320kbps, 192, 24bit, 16bit, V0, etc.)
- **Resolution**: Patterns (1080p, 720p, 4K, 2160p, etc.)
- **Audio languages**: Patterns (English, ENG, MULTI, DUAL, Spanish, etc.)
- **Subtitle languages**: Patterns (SUBS, English.srt, SPA.SUBS, embedded patterns, etc.)

Note: Subtitle languages inferred from both:
- External subtitle files (.srt, .sub, .ass)
- Filename patterns indicating embedded subs (MULTI.SUBS, ENG.SPA.SUBS, etc.)

### Content Metadata Model Options

1. **LLM labeling → train smaller model**: Use LLM with JSON schema prompting to label dataset, then fine-tune a small model for extraction
2. **Direct LLM inference**: Use small LLM (Qwen2.5-0.5B) embedded in binary for extraction
3. **Hybrid**: Regex extraction for common patterns, LLM for ambiguous cases

## Implementation Order

1. [ ] Content metadata labeling pipeline (LLM)
2. [ ] Train/integrate content metadata model
3. [ ] Technical metadata extraction (regex on remainder, in Rust)

## Open Questions

- [ ] For audio/collection, how to handle Various Artists compilations? (Current approach: `artists[]` array can hold multiple artists or `["Various Artists"]`)

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
   - Technical: resolution, codec, bitrate, containers, audio_languages, subtitle_languages
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

---

## Ground-Truth Training Data Generation (2025-01-20)

### The Problem with LLM Labeling

LLM consensus extraction (qwen3-coder + gpt-oss-120b) achieved only ~52% agreement on random torrent samples. This means:
- 48% of samples have ambiguous or contested labels
- No ground truth to verify which model is correct
- Training on contested labels introduces noise

### New Approach: Reverse Lookup from Known Metadata

Instead of extracting metadata from random torrents, we **start with known-good metadata** and find matching torrents:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GROUND-TRUTH PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │   TMDB      │    │ MusicBrainz │    │  Other DBs  │            │
│   │ (movies,TV) │    │  (albums)   │    │  (future)   │            │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
│          │                  │                  │                    │
│          └──────────────────┼──────────────────┘                    │
│                             ▼                                       │
│                  ┌─────────────────────┐                            │
│                  │   Ground Truth DB   │                            │
│                  │   (title, year,     │                            │
│                  │    artist, type)    │                            │
│                  └──────────┬──────────┘                            │
│                             │                                       │
│                             ▼                                       │
│                  ┌─────────────────────┐                            │
│                  │  Magnetico Search   │                            │
│                  │  (FTS5 full-text)   │                            │
│                  │  32.5M torrents     │                            │
│                  └──────────┬──────────┘                            │
│                             │                                       │
│                             ▼                                       │
│                  ┌─────────────────────┐                            │
│                  │  Heuristic Scoring  │                            │
│                  │  - Title match      │                            │
│                  │  - Year match       │                            │
│                  │  - Size sanity      │                            │
│                  └──────────┬──────────┘                            │
│                             │                                       │
│                             ▼                                       │
│                  ┌─────────────────────┐                            │
│                  │  Training Samples   │                            │
│                  │  Input: torrent     │                            │
│                  │  Label: ground truth│                            │
│                  └─────────────────────┘                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Sources

| Source | Content Type | Target Count | Expected Matches |
|--------|--------------|--------------|------------------|
| TMDB | Movies | 2,000 | ~6,000 (3/movie) |
| TMDB | TV Series | 500 | ~1,500 (3/series) |
| MusicBrainz | Albums | 5,000 | ~10,000 (2/album) |
| **Total** | | **7,500** | **~17,500 samples** |

### Implementation Steps

1. **Fetch ground truth metadata**
   - TMDB API: `/movie/top_rated`, `/tv/top_rated` (paginated)
   - MusicBrainz API: popular releases query
   - Store in SQLite: `ground_truth(id, type, title, artist, year, tmdb_id, mb_id)`

2. **Search Magnetico for matches**
   - FTS5 query: `"Inception 2010"` for movies
   - FTS5 query: `"Pink Floyd" "Dark Side"` for albums
   - Store in: `matches(ground_truth_id, torrent_id, torrent_name, score)`

3. **Score and filter matches**
   - **Title match**: Levenshtein distance or fuzzy match
   - **Year match**: Exact match in torrent name (+5 points)
   - **Size sanity**: Movies 700MB-50GB, Albums 50MB-2GB
   - **Negative signals**: "sample", "trailer", "teaser" (-10 points)
   - Keep matches with score > threshold

4. **Generate training samples**
   ```json
   {
     "input": "Inception.2010.1080p.BluRay.x264-SPARKS",
     "label": {
       "type": "video/movie",
       "title": "Inception",
       "year": 2010
     }
   }
   ```

5. **(Optional) Synthetic augmentation**
   - Generate naming variations from real matches
   - `Inception (2010) 1080p` → `Inception.2010.1080p`, `[YTS] Inception 2010`
   - Can 3-5x the dataset size

### Advantages

| Aspect | LLM Consensus | Ground-Truth Lookup |
|--------|---------------|---------------------|
| Label accuracy | ~52% agreement | 100% (by definition) |
| Scalability | Slow (LLM inference) | Fast (SQL queries) |
| Coverage | Random samples | Popular content |
| Augmentation | Limited | Easy synthetic variants |

### Scripts

- `scripts/fetch_tmdb_metadata.py` - Pull movies/series from TMDB
- `scripts/fetch_musicbrainz_metadata.py` - Pull albums from MusicBrainz
- `scripts/match_torrents.py` - Search magnetico, score matches
- `scripts/export_training_data.py` - Generate final training set

### Database Schema

```sql
-- Ground truth from external sources
CREATE TABLE ground_truth (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,  -- 'movie', 'tv', 'album'
    title TEXT NOT NULL,
    artist TEXT,         -- for albums
    year INTEGER,
    external_id TEXT,    -- TMDB ID or MusicBrainz ID
    source TEXT NOT NULL -- 'tmdb' or 'musicbrainz'
);

-- Matched torrents from magnetico
CREATE TABLE matches (
    id INTEGER PRIMARY KEY,
    ground_truth_id INTEGER REFERENCES ground_truth(id),
    torrent_name TEXT NOT NULL,
    torrent_size INTEGER,
    match_score REAL,
    UNIQUE(ground_truth_id, torrent_name)
);

-- Final training samples
CREATE TABLE training_samples (
    id INTEGER PRIMARY KEY,
    input TEXT NOT NULL,      -- torrent name (+ files)
    label_json TEXT NOT NULL, -- ground truth as JSON
    source TEXT NOT NULL,     -- 'matched' or 'synthetic'
    match_id INTEGER REFERENCES matches(id)
);
```

---

## Album Metadata Extraction: Heuristics Approach (2025-01-21)

### Background

After evaluating GLiNER for album metadata extraction, results were not acceptable. Two options were considered:
1. Fine-tune a small LLM that can run on CPU
2. Double down on heuristics, leveraging the large ground-truth dataset to iteratively refine rules

**Decision**: Try heuristics first since we have a massive labeled dataset (~14k album samples from MusicBrainz matches).

### Approach: Data-Driven Heuristic Refinement

Instead of writing heuristics upfront, we use the ground-truth dataset to "train" the heuristics iteratively:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 HEURISTIC REFINEMENT LOOP                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                  │
│   │    TODO     │  Samples that don't match yet                    │
│   │   (5105)    │                                                  │
│   └──────┬──────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────┐                                          │
│   │  Run Heuristics     │                                          │
│   │  extract(name) →    │                                          │
│   │  {artist,title,year}│                                          │
│   └──────────┬──────────┘                                          │
│              │                                                      │
│              ▼                                                      │
│   ┌─────────────────────┐                                          │
│   │  Compare to Ground  │  Year: exact match                       │
│   │  Truth              │  Artist/Title: ≥70% similarity           │
│   │                     │  (using MusicBrainz aliases)             │
│   └──────────┬──────────┘                                          │
│              │                                                      │
│       ┌──────┴──────┐                                              │
│       ▼             ▼                                              │
│   ┌───────┐    ┌───────┐                                           │
│   │ MATCH │    │ FAIL  │                                           │
│   │ → DONE│    │→ TODO │                                           │
│   └───────┘    └───────┘                                           │
│                    │                                                │
│                    ▼                                                │
│          ┌─────────────────────┐                                   │
│          │  Analyze failures,  │                                   │
│          │  update heuristics  │                                   │
│          └──────────┬──────────┘                                   │
│                     │                                               │
│                     ▼                                               │
│          ┌─────────────────────┐                                   │
│          │  REGRESSION TEST    │  Run on DONE items               │
│          │  All must still pass│  If any fail, revert changes     │
│          └─────────────────────┘                                   │
│                                                                     │
│   Repeat until plateau or ≥80% success rate                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Success Criteria

- **Year**: Exact match required
- **Artist**: ≥70% string similarity (using SequenceMatcher), checked against MusicBrainz aliases
- **Title**: ≥70% string similarity
- **Target**: ≥80% of samples passing = heuristics approach is viable

### Current Progress

| Version | Done | Todo | Success Rate |
|---------|------|------|--------------|
| v1 (baseline) | 8,151 | 5,663 | 59.0% |
| v2 (improved year extraction) | 8,417 | 5,397 | 60.9% |
| v3 (safe release group removal, editions) | 8,709 | 5,105 | **63.0%** |

### Key Heuristics (v3)

1. **Year extraction**: Multiple patterns - `(2024)`, `[2024]`, `[2024.11.22]`, standalone year
2. **Format tag removal**: FLAC, MP3, 320kbps, 24bit, WEB, CD, etc.
3. **Release group removal**: Whitelist-based (EICHBAUM, JLM, OMA, etc.) instead of generic pattern
4. **Edition stripping**: "(with Isolated Vocals)", "(Store Exclusive)", "(00XO Edition)"
5. **Scene format cleanup**: Handle concatenation artifacts like `-FLACJLM`, `-ESOMA`
6. **Artist-Album split**: `^(.+?)\s*[-–—]\s*(.+)$` pattern

### MusicBrainz Artist Aliases

Extracted 9,480 aliases for artists in our ground truth from the MusicBrainz dump (`~/Downloads/musicbrainz/artist.tar.xz`). This helps match variations like:
- "The Weeknd" ↔ "Abel Tesfaye"
- "50 Cent" ↔ "Curtis Jackson"

### Fallback Plan

If heuristics plateau below 80%, fall back to:
- Fine-tune a small LLM (e.g., Qwen2.5-0.5B) on the labeled dataset
- Use the heuristics-extracted samples as additional training data

### Scripts

- `scripts/album_heuristics.py` - Main heuristics pipeline (run, regress, stats, failures)
- `scripts/extract_mb_aliases.py` - Extract artist aliases from MusicBrainz dump
- `scripts/fetch_musicbrainz_metadata.py` - Fetch ground truth from MusicBrainz API

### Database Tables

```sql
-- Heuristics iteration tracking
CREATE TABLE todo (
    id INTEGER PRIMARY KEY,
    torrent_name TEXT NOT NULL,
    expected_artist TEXT,
    expected_title TEXT,
    expected_year INTEGER,
    ground_truth_id INTEGER
);

CREATE TABLE done (
    id INTEGER PRIMARY KEY,
    torrent_name TEXT NOT NULL,
    expected_artist TEXT,
    expected_title TEXT,
    expected_year INTEGER,
    ground_truth_id INTEGER,
    extracted_artist TEXT,
    extracted_title TEXT,
    extracted_year INTEGER,
    heuristic_version INTEGER
);

CREATE TABLE artist_aliases (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    alias TEXT NOT NULL,
    mb_id TEXT,
    UNIQUE(canonical_name, alias)
);
```
