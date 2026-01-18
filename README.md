# MIMMO

**Media Intelligence Metadata Modeling Outlet**

A Rust library and CLI for torrent content classification. Classifies torrents into media types (audio, video, software, book, other) with subcategory detection for audio/video content.

Single binary, no dependencies, CPU inference via embedded ONNX model.

## Installation

```bash
cargo install mimmo
```

Or build from source:
```bash
git clone https://github.com/lelloman/mimmo
cd mimmo
cargo build --release
```

## CLI Usage

```bash
# Classify a torrent file
mimmo /path/to/file.torrent

# Classify a directory
mimmo /path/to/extracted/torrent/

# Classify an archive
mimmo album.zip
mimmo release.tar.gz

# Classify raw text
mimmo "Pink Floyd - Dark Side of the Moon [FLAC]"

# Pipe input
echo "Game of Thrones S01 Complete 1080p" | mimmo
```

Output is JSON:
```json
{"medium":"audio","subcategory":"album","confidence":0.9706}
{"medium":"video","subcategory":"season","confidence":0.9512}
{"medium":"software","confidence":0.8834}
```

## Library Usage

```rust
use mimmo::{Classifier, from_torrent, from_path};

let mut classifier = Classifier::new()?;

// From torrent file
let info = from_torrent("/path/to/file.torrent")?;
let result = classifier.classify(&info)?;

// From any supported path (auto-detects type)
let info = from_path("/path/to/content")?;
let result = classifier.classify(&info)?;

println!("{} ({:.1}%)", result.medium, result.confidence * 100.0);
if let Some(sub) = result.subcategory {
    println!("Subcategory: {}", sub);
}
```

## Classification

### Medium Types

| Medium | Description |
|--------|-------------|
| `audio` | Music albums, tracks, discographies |
| `video` | Movies, TV series, documentaries |
| `software` | Applications, games, operating systems |
| `book` | Ebooks, PDFs, comics |
| `other` | Unclassified content |

### Subcategories

For audio and video content, structural analysis determines subcategory:

**Audio:**
| Subcategory | Detection |
|-------------|-----------|
| `track` | Single audio file |
| `album` | Multiple audio files in one directory |
| `collection` | Multiple audio files across multiple directories |

**Video:**
| Subcategory | Detection |
|-------------|-----------|
| `movie` | Single video file (no episode pattern) |
| `episode` | Single video file with S01E01/1x01 pattern |
| `season` | Multiple video files in one directory |
| `series` | Multiple video files across multiple directories |

## Supported Input Formats

- Torrent files (`.torrent`)
- Directories
- Zip archives (`.zip`, `.cbz`, `.epub`, `.jar`, `.apk`)
- Tar archives (`.tar`, `.tar.gz`, `.tgz`, `.tar.xz`, `.txz`)
- Raw text (torrent name)

## Model

The classifier uses a fine-tuned BERT-tiny model (~4MB) embedded in the binary:
- Base: `prajjwal1/bert-tiny`
- Training: ~10k samples with 4-LLM consensus voting
- Accuracy: ~92% on held-out test set
- Inference: <10ms per sample on CPU

## Project Status

### Completed
- [x] Medium classification (audio/video/software/book/other)
- [x] Subcategory detection (album/track/collection, movie/episode/season/series)
- [x] Rust library with public API
- [x] CLI binary
- [x] ONNX model inference

### Planned
- [ ] Metadata extraction (title, artist, year, etc.)
- [ ] Technical metadata (codec, bitrate, resolution)

## Repository Structure

```
mimmo/
├── src/
│   ├── lib.rs              # Library with public API
│   └── main.rs             # CLI binary
├── training/
│   ├── consensus_labeler.py    # 4-LLM consensus labeling
│   ├── train_classifier.py     # BERT training script
│   └── bert-classifier-medium/ # Trained model
├── Cargo.toml
└── README.md
```

## Training Data

Content classifier training data is generated using multi-LLM consensus voting:

**Models:** qwen2.5:3b, gemma3:4b, mistral:7b, qwen3-coder:30b

**Consensus rules:**
- All 4 agree (~60%) = high confidence label
- 3v1 majority (~27%) = majority vote label
- 2v2 split = discarded

**Data source:** [Magnetico DHT dump](https://tnt.maiti.info/dhtd/) (32.5M torrents)

## License

MIT
