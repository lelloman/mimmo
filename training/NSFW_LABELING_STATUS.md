# NSFW Labeling Status

**Status:** COMPLETE

## Summary

NSFW classifier trained and deployed. Model available at [lelloman/bert-torrent-nsfw](https://huggingface.co/lelloman/bert-torrent-nsfw).

## Training Data

| Model | Samples | Notes |
|-------|---------|-------|
| gemma (gemma3-4b) | 50,000 | Complete |
| qwen (Qwen3-Coder-30B) | ~10,000 | Partial |

Final training set: ~10k samples with consensus labeling.

## Deployment

- Model: BERT-tiny binary classifier (~17MB ONNX)
- Location: `models/bert-nsfw/model_embedded.onnx`
- HuggingFace: [lelloman/bert-torrent-nsfw](https://huggingface.co/lelloman/bert-torrent-nsfw)

## Usage

```bash
mimmo --detect-nsfw "Torrent.Name.Here"
```

Three-stage cascade:
1. **Keywords** - HashSet lookup for explicit terms (confidence: 0.95)
2. **Patterns** - Regex for JAV codes, studio patterns (confidence: 0.90)
3. **ML** - BERT classifier fallback (variable confidence)
