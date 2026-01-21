# MIMMO

**Media Intelligence Metadata Modeling Outlet**

A Rust library and CLI for torrent content classification and metadata extraction. Classifies torrents into media types (audio, video, software, book, other) with subcategory detection and extracts structured metadata (title, artist, year) for audio/video content.

Single binary (~309MB), no runtime dependencies, CPU inference via embedded ONNX + GGUF models.

## Installation

Build from source:
```bash
git clone https://github.com/lelloman/mimmo
cd mimmo
cargo build --release
```

The build script automatically downloads both models (~276MB total) from HuggingFace on first build.

## CLI Usage

```bash
# Classify and extract metadata
mimmo "Pink Floyd - The Dark Side of the Moon (1973) [FLAC]"
# {"medium":"audio","subcategory":"album","confidence":0.95,"source":"patterns","metadata":{"title":"The Dark Side Of The Moon","artist":"Pink Floyd","year":1973}}

mimmo "The.Matrix.1999.1080p.BluRay.x264-GROUP"
# {"medium":"video","subcategory":"movie","confidence":0.95,"source":"patterns","metadata":{"title":"The Matrix","year":1999}}

# Classify a torrent file
mimmo /path/to/file.torrent

# Classify a directory
mimmo /path/to/extracted/torrent/

# Classify an archive
mimmo album.zip
mimmo release.tar.gz

# Pipe input
echo "Game of Thrones S01 Complete 1080p" | mimmo

# Interactive batch mode (JSON arrays)
mimmo -i
["torrent1", "torrent2", "torrent3"]
# [{"medium":"audio",...},{"medium":"video",...},...]
```

## Library Usage

Add mimmo to your `Cargo.toml`:
```toml
[dependencies]
mimmo = { git = "https://github.com/lelloman/mimmo" }
```

Example:
```rust
use mimmo::{from_path, from_text, detect_subcategory};
use mimmo::cascade::Cascade;
use mimmo::metadata::MetadataExtractor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load content from various sources
    let info = from_path("/path/to/file.torrent")?;
    // or: from_text("Pink Floyd - The Dark Side of the Moon (1973) [FLAC]")

    // Classify using cascade (patterns + ML fallback)
    let cascade = Cascade::default_with_ml()?;
    let result = cascade.classify(&info)?;

    println!("Medium: {:?}", result.medium);       // Audio, Video, Software, Book, Other
    println!("Confidence: {:?}", result.confidence); // High, Medium, Low
    println!("Source: {}", result.source);          // "extension", "patterns", "ml"

    // Detect subcategory for audio/video
    let subcategory = detect_subcategory("audio", &info); // album, track, collection

    // Extract metadata for audio/video content
    let extractor = MetadataExtractor::new()?;
    let meta = extractor.extract(&info.name, "audio/album")?;

    println!("Title: {}", meta.title);
    println!("Artist: {:?}", meta.artist);
    println!("Year: {:?}", meta.year);

    Ok(())
}
```

## Output Format

```json
{
  "medium": "audio",
  "subcategory": "album",
  "confidence": 0.95,
  "source": "patterns",
  "metadata": {
    "title": "The Dark Side Of The Moon",
    "artist": "Pink Floyd",
    "year": 1973
  }
}
```

The `metadata` field is only present for audio/video content.

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

## Models

### Content Classifier
- BERT-tiny model (~17MB ONNX) embedded in binary
- Base: `prajjwal1/bert-tiny`
- Training: ~10k samples with 4-LLM consensus voting
- Accuracy: ~92% on held-out test set
- Model: [lelloman/bert-torrent-classifier](https://huggingface.co/lelloman/bert-torrent-classifier)
- Inference: <10ms per sample

### Metadata Extractor
- SmolLM-360M fine-tuned for metadata extraction (~259MB Q4_K_M)
- Base: `HuggingFaceTB/SmolLM-360M-Instruct`
- Training: ~200k samples from Spotify catalog validation
- Model: [lelloman/smollm-torrent-metadata](https://huggingface.co/lelloman/smollm-torrent-metadata)
- Inference: ~250ms per sample (CPU)

## Building

```bash
cargo build --release
```

The build script automatically downloads both models from HuggingFace on first build:
- BERT classifier (~17MB) from [lelloman/bert-torrent-classifier](https://huggingface.co/lelloman/bert-torrent-classifier)
- SmolLM metadata extractor (~259MB) from [lelloman/smollm-torrent-metadata](https://huggingface.co/lelloman/smollm-torrent-metadata)

Both models are embedded in the binary at compile time using `include_bytes!`.

## Repository Structure

```
mimmo/
├── src/
│   ├── lib.rs              # Library with public API
│   ├── main.rs             # CLI binary
│   ├── cascade/            # Classification cascade
│   └── metadata.rs         # SmolLM metadata extraction
├── scripts/
│   ├── train_smollm.py     # LLM fine-tuning
│   ├── prepare_llm_training_data.py
│   └── validate_spotify.py # Training data validation
├── training/
│   ├── train_classifier.py # BERT classifier training
│   └── convert_to_onnx.py
├── models/
│   └── gguf/               # GGUF model files (gitignored)
├── Cargo.toml
└── README.md
```

## Training Pipeline

### Content Classifier
Training data generated using multi-LLM consensus voting:
- **Models:** qwen2.5:3b, gemma3:4b, mistral:7b, qwen3-coder:30b
- **Consensus rules:** 4-agree = high confidence, 3v1 = majority, 2v2 = discarded
- **Data source:** [Magnetico DHT dump](https://tnt.maiti.info/dhtd/) (32.5M torrents)

### Metadata Extractor
- Fine-tuned SmolLM-360M with LoRA (r=16, alpha=32)
- Training data from Spotify catalog reverse-matching
- GGUF quantization via llama.cpp

## License

MIT
