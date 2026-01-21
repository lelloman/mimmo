# NSFW Detection - Implementation Plan

**Last Updated:** 2026-01-21
**Status:** Ready for implementation

---

## Overview

Add NSFW detection as an optional feature via `--detect-nsfw` flag. NSFW is **orthogonal** to media classification - a torrent can be `video/movie` AND `nsfw: true`.

---

## Architecture Decision

### Chosen Approach: Separate BERT-tiny + Shared Tokenizer

| Component | Current (medium) | New (NSFW) |
|-----------|------------------|------------|
| Model | BERT-tiny (~4MB ONNX) | BERT-tiny (~2-4MB ONNX) |
| Tokenizer | `prajjwal1/bert-tiny` | **Shared** (already embedded) |
| Task | 5-class classification | Binary classification |
| Weights | `bert-classifier-medium/onnx/model.onnx` | `bert-classifier-nsfw/onnx/model.onnx` |

**Why separate models:**
- NSFW is a flag, not a category (orthogonal to medium/subcategory)
- Binary classification = smaller weights than 5-class
- Zero overhead when `--detect-nsfw` not passed
- Independent training/updates without affecting medium classifier

**Why shared tokenizer:**
- Same base model (`prajjwal1/bert-tiny`) = same vocabulary
- Tokenizer already embedded (~2MB) - no additional size
- Load once at runtime, use for both models

---

## Detection Cascade

Similar to medium classification, use a cascade for efficiency:

```
--detect-nsfw flow:

Input name
    ↓
[Stage 1: Keywords] ─→ Explicit terms match? → nsfw: true, confidence: high
    ↓ (no match)
[Stage 2: Patterns] ─→ Porn filename patterns? → nsfw: true, confidence: high
    ↓ (no match)
[Stage 3: BERT-tiny-NSFW] ─→ ML inference → nsfw: true/false, confidence: model score
```

### Stage 1: Keyword List

Fast first pass for obvious terms. Include:
- Explicit terms: `xxx`, `porn`, `adult`, `nsfw`, `hentai`
- Studio names (common porn studios)
- Common abbreviations

**Implementation:** `HashSet<&'static str>` lookup - O(1)

### Stage 2: Pattern Matching

Detect common porn filename structures:
- Prefixes: `xxx.`, `[xxx]`, `porn.`
- Scene naming patterns: `performer1.performer2.studio.resolution`
- Date patterns common in porn releases

**Implementation:** Regex patterns in `src/cascade/stages/nsfw_patterns.rs`

### Stage 3: BERT-tiny NSFW Model

For ambiguous cases - slang, obfuscated terms, context-dependent names.

**Implementation:** Same architecture as `src/cascade/stages/ml.rs`, separate weights file.

---

## Output Format

```json
{
  "medium": "video",
  "subcategory": "movie",
  "confidence": 0.92,
  "source": "patterns",
  "nsfw": true,
  "nsfw_confidence": 0.95,
  "nsfw_source": "keywords",
  "metadata": {
    "title": "...",
    "year": 2024
  }
}
```

When `--detect-nsfw` not passed, `nsfw*` fields are omitted.

---

## Training Pipeline

Reuse existing infrastructure with modifications:

### 1. Data Labeling (`training/nsfw_labeler.py`)

Adapt `consensus_labeler.py` for binary NSFW classification:

```python
CATEGORIES = ["safe", "nsfw"]

PROMPT = """Is this torrent NSFW/adult content? Reply with exactly one word: safe or nsfw

Torrent: {name}
Top files:
{files}

Classification:"""
```

**LLM consensus:** Same 4-model voting (qwen2.5:3b, gemma3:4b, mistral:7b, qwen3-coder:30b)

**Data source:**
- Sample from existing DHT dump (32.5M torrents)
- Oversample from "porn" category in existing `consensus_labels.db` for positive examples
- Need ~10-20k samples (binary is easier than 5-class)

### 2. Model Training (`training/train_nsfw_classifier.py`)

Adapt `train_classifier.py`:

```python
MODEL_NAME = "prajjwal1/bert-tiny"  # Same base model
NUM_LABELS = 2

LABEL2ID = {"safe": 0, "nsfw": 1}
ID2LABEL = {0: "safe", 1: "nsfw"}
```

**Hyperparameters:** Same as medium classifier (5 epochs, batch 64, lr 5e-5)

### 3. ONNX Export (`training/convert_to_onnx.py`)

Same process - export to `bert-classifier-nsfw/onnx/model.onnx`

**Note:** Only export model weights, not tokenizer (shared with medium classifier)

---

## Rust Implementation

### New Files

```
src/cascade/stages/nsfw.rs       # NSFW cascade (keywords + patterns + ML)
src/nsfw_keywords.rs             # Keyword list (HashSet)
src/nsfw_patterns.rs             # Regex patterns
```

### Modified Files

```
src/lib.rs                       # Add nsfw model weights, NsfwResult struct
src/main.rs                      # Add --detect-nsfw flag
src/cascade/mod.rs               # Optional NSFW detection
```

### Key Structs

```rust
// In src/lib.rs
pub struct NsfwResult {
    pub nsfw: bool,
    pub confidence: f32,
    pub source: NsfwSource,
}

pub enum NsfwSource {
    Keywords,
    Patterns,
    Ml,
}

// Updated ClassificationResult
pub struct ClassificationResult {
    pub medium: Medium,
    pub subcategory: Option<Subcategory>,
    pub confidence: f32,
    pub source: Source,
    pub nsfw: Option<NsfwResult>,  // Only present when --detect-nsfw
    pub metadata: Option<Metadata>,
}
```

### CLI Interface

```bash
# Without NSFW detection (default)
mimmo classify "Some.Movie.2024.1080p.mkv"
# {"medium":"video","subcategory":"movie",...}

# With NSFW detection
mimmo classify --detect-nsfw "Some.Movie.2024.1080p.mkv"
# {"medium":"video","subcategory":"movie",...,"nsfw":false,"nsfw_confidence":0.92}
```

---

## Size Impact

| Component | Size |
|-----------|------|
| Current binary | ~309 MB |
| NSFW model weights (ONNX) | ~2-4 MB |
| Keyword list | ~10 KB |
| Pattern regexes | ~5 KB |
| **Total increase** | **~2-4 MB** |

Tokenizer is shared - no additional size.

---

## Implementation Steps

### Phase 1: Training Data
1. [ ] Create `training/nsfw_labeler.py` (adapt from consensus_labeler.py)
2. [ ] Sample ~20k torrents, with oversampling from known porn category
3. [ ] Run 4-LLM consensus labeling
4. [ ] Export training data to JSONL

### Phase 2: Model Training
5. [ ] Create `training/train_nsfw_classifier.py` (adapt from train_classifier.py)
6. [ ] Train BERT-tiny binary classifier
7. [ ] Export to ONNX (weights only, no tokenizer)
8. [ ] Validate accuracy (target: >95% for binary)

### Phase 3: Rust Integration
9. [ ] Add keyword list (`src/nsfw_keywords.rs`)
10. [ ] Add pattern matching (`src/nsfw_patterns.rs`)
11. [ ] Add NSFW cascade stage (`src/cascade/stages/nsfw.rs`)
12. [ ] Embed NSFW model weights in binary
13. [ ] Add `--detect-nsfw` CLI flag
14. [ ] Update output structs and JSON serialization

### Phase 4: Testing
15. [ ] Unit tests for keywords/patterns
16. [ ] Integration tests for full cascade
17. [ ] Benchmark inference speed
18. [ ] Manual testing with real torrent names

---

## Research References (Historical)

Previous research evaluated these approaches before deciding on BERT-tiny:

| Tool | Size | Speed | Accuracy | Decision |
|------|------|-------|----------|----------|
| better-profanity | <1 MB | Fastest | Low | Too limited |
| profanity-check | ~5-10 MB | Fast | Medium | Not Rust-native |
| DistilBERT NSFW | ~268 MB | Slow | High | Too large |
| **BERT-tiny** | **~4 MB** | **<10ms** | **High** | **Chosen** |

Key papers reviewed:
- [Learning Strategies for Sensitive Content Detection](https://www.mdpi.com/2079-9292/12/11/2496) (2023)
- [NSFW Text Identification](https://www.researchgate.net/publication/364652449_NSFW_Text_Identification) (2022)
