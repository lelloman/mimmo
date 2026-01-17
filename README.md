# MIMMO

**Media Intelligence Metadata Modeling Outlet**

A two-stage pipeline for torrent content classification and music metadata extraction:
1. **Content Classifier** (BERT) - Fast content type detection
2. **Music Extractor** (LLM) - Detailed metadata extraction for music
3. **Name Matcher** (LLM) - Fuzzy artist/album matching

Single binary, no dependencies, CPU inference.

## Architecture

```
                    ┌─────────────────┐
                    │  Torrent Input  │
                    │  name + files   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  BERT Classifier│  ← Stage 1: Content Type
                    │  (~4MB, <1ms)   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    ┌───────┐           ┌───────┐           ┌───────┐
    │ music │           │ movie │           │ other │
    └───┬───┘           └───────┘           └───────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐       (skip)              (skip)
│ LLM Extractor │  ← Stage 2: Metadata
│ Qwen2.5-0.5B  │
└───────────────┘
        │
        ▼
┌───────────────┐
│ {artist, album│
│  year, tracks}│
└───────────────┘
```

## CLI Interface

```bash
# Classify content type (Stage 1 only)
mimmo content-type "Pink Floyd - Dark Side of the Moon [FLAC]"
{"type": "music", "confidence": 0.98}

# Full extraction (Stage 1 + Stage 2)
mimmo extract < tree.txt
{
  "content_type": "music",
  "music_type": "album",
  "artist": "Pink Floyd",
  "album": "The Dark Side of the Moon",
  "year": 1973,
  "tracks": [
    {"num": 1, "name": "Speak to Me"},
    {"num": 2, "name": "Breathe"},
    ...
  ]
}

# Name matching
mimmo match artist "GNR" "Guns N' Roses"
{"match": true, "confidence": 0.92}

mimmo match album "DSOTM" "The Dark Side of the Moon"
{"match": true, "confidence": 0.96}
```

## Models

| Model | Task | Base | Size | Latency |
|-------|------|------|------|---------|
| Content Classifier | music/video/software/etc | BERT-tiny | ~4MB | <1ms |
| Music Extractor | artist/album/year/tracks | Qwen2.5-0.5B | ~300MB | <1s |
| Name Matcher | artist/album fuzzy match | Qwen2.5-0.5B | (shared) | <200ms |

## Content Types

The classifier detects:
- `music` - Albums, discographies, singles
- `video` - Films, TV series, documentaries, anime
- `software` - Applications, games, tools
- `book` - Ebooks, audiobooks, comics
- `porn` - Adult content
- `other` - Unclassified

## Project Phases

### Phase 1: Data Collection ✅
- [x] Download Magnetico DHT dump (32.5M torrents)
- [x] Download MusicBrainz JSON dumps (artist + release-group)

### Phase 2: Training Data Preparation
- [ ] **Content Classifier Data**
  - [ ] Sample 50k torrents from DHT dump
  - [ ] Label with 4 LLM consensus voting (qwen2.5:3b, gemma3:4b, mistral:7b, qwen3-coder:30b)
  - [ ] All 4 agree (~60%) = high confidence, 3v1 (~27%) = majority vote
  - [ ] Output: `training_data_consensus.jsonl` (~43k usable samples)
- [ ] **Music Metadata Data**
  - [ ] Filter music torrents (audio file extensions)
  - [ ] Label with LLM: artist, album, year, tracks
  - [ ] Output: `music_metadata.jsonl` (~50k samples)
- [ ] **Name Matching Data** ✅
  - [x] Extract MusicBrainz artist aliases → `artist_pairs.jsonl` (10k)
  - [x] Extract MusicBrainz album variations → `album_pairs.jsonl` (10k)

### Phase 3: Model Training
- [ ] **BERT Content Classifier**
  - [ ] Fine-tune `prajjwal1/bert-tiny` or `distilbert-base-uncased`
  - [ ] Target: >95% accuracy on content type
  - [ ] Export to ONNX or Safetensors
- [ ] **LLM Music Extractor**
  - [ ] Fine-tune Qwen2.5-0.5B with LoRA
  - [ ] Constrained JSON output
  - [ ] Target: >90% field accuracy
- [ ] **LLM Name Matcher** (optional separate training)
  - [ ] May share weights with extractor

### Phase 4: Rust Binary
- [ ] Integrate BERT inference (candle or ort)
- [ ] Integrate LLM inference (candle)
- [ ] CLI implementation
- [ ] Single binary packaging
- [ ] Cross-compilation

## Training Data Sources

| Source | Data | Use |
|--------|------|-----|
| [Magnetico DHT dump](https://tnt.maiti.info/dhtd/) | 32.5M torrents | Content classification, music extraction |
| [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_Database/Download) | Artist/album data | Name matching pairs |

## Repository Structure

```
mimmo/
├── README.md
├── training/
│   ├── consensus_labeler.py        # 4-LLM consensus labeling for BERT data
│   ├── extract_artist_pairs.py     # MusicBrainz → artist pairs
│   ├── extract_album_pairs.py      # MusicBrainz → album pairs
│   ├── extract_music_torrents.py   # DHT → music samples
│   ├── extract_content_samples.py  # DHT → content type samples
│   ├── train_classifier.py         # BERT training (TODO)
│   └── train_extractor.py          # LLM fine-tuning (TODO)
├── data/                           # Training data (gitignored)
├── prompts/                        # LLM prompt templates
├── ai_torrent_analyzer.py          # Manual testing tool
├── dht_search.py                   # DHT database search
└── dht_search_tui.py               # TUI for DHT search
```

## Labeling Infrastructure

The content classifier training data is generated using multi-LLM consensus voting:

**Models:**
- 3 small models on RTX 4090 via Ollama: qwen2.5:3b, gemma3:4b, mistral:7b
- 1 large model on Strix Halo via OpenAI API: qwen3-coder:30b

**Performance (benchmarked on 1000 samples):**
- RTX Ollama: ~31 req/s with 4 workers
- Strix Halo OpenAI: ~4 req/s
- All 4 agree: ~60% of samples
- 3v1 majority: ~27% additional samples
- Total usable: ~87% of samples

**Usage:**
```bash
# Full run: sample and label with all 4 models
python training/consensus_labeler.py -n 50000

# Run 3 small models only (RTX)
python training/consensus_labeler.py --skip-sampling --model small3

# Run big model only (Strix Halo)
python training/consensus_labeler.py --skip-sampling --model qwen3coder

# Check progress
python training/consensus_labeler.py --stats

# Export training data
python training/consensus_labeler.py --export
```

## License

MIT
