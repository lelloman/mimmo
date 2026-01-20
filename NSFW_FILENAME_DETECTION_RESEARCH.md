# NSFW Content Detection from File Names - Research Summary

**Date:** 2025-01-21
**Purpose:** ML tools and approaches for detecting pornographic content based on file names

---

## Overview

Detecting NSFW content from file names requires text-based classification approaches rather than image analysis. This document summarizes available tools, libraries, and strategies.

---

## Available ML Tools & Libraries

### 1. DistilBERT NSFW Text Classifier

- **Model:** `eliasalbouzidi/distilbert-nsfw-text-classifier` on Hugging Face
- **Size:** ~268 MB (model.safetensors)
- **Type:** Transformer-based binary classification
- **Classes:** "safe" vs "nsfw"
- **Use case:** High accuracy, handles slang and context
- **Trade-off:** Large file size, slower inference
- **Link:** https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier

### 2. Profanity-Check

- **Package:** `profanity-check` on PyPI
- **Size:** ~5-10 MB
- **Type:** SVM model trained on comments
- **Use case:** Good balance of accuracy and speed
- **Trade-off:** Less accurate on slang compared to transformers
- **Link:** https://pypi.org/project/profanity-check/

### 3. Better-Profanity

- **Package:** `better-profanity` on PyPI
- **Size:** <1 MB (keyword list only)
- **Type:** Pure keyword matching
- **Use case:** Fastest option, simple implementation
- **Trade-off:** Limited coverage, no ML understanding
- **Link:** https://github.com/snguyenthanh/better_profanity

### 4. Profanity-Filter

- **Package:** `profanity-filter` on PyPI
- **Type:** Multi-language profanity detection
- **Features:** Censoring, custom word lists
- **Link:** https://pypi.org/project/profanity-filter/

---

## API Services

### Replicate NSFW Detection
- **Link:** https://replicate.com/collections/detect-nsfw-content
- Multiple models available for text and image detection
- Pay-per-use, no local model needed

---

## Academic Research

### Key Papers

1. **[Learning Strategies for Sensitive Content Detection](https://www.mdpi.com/2079-9292/12/11/2496)** (2023)
   - Covers textual features including filenames, keywords, metadata
   - Multi-modal approach recommendations

2. **[NSFW Text Identification](https://www.researchgate.net/publication/364652449_NSFW_Text_Identification)** (2022)
   - Research on NSFW text classification

3. **[Pipeline Using Filenames and Metadata](https://www.researchgate.net/figure/Pipeline-of-sensitive-content-classification-using-filenames-and-metadata-features_fig4_371262358)**
   - CNN models using filename features for classification

---

## Implementation Strategy

### Recommended Hybrid Approach

For effective filename-based NSFW detection, combine multiple methods:

1. **Keyword List** - Fast first pass for obvious terms
2. **Pattern Matching** - Detect common porn filename structures:
   - Repeated keywords
   - Specific prefixes (e.g., "xxx", "porn", "adult")
   - Number patterns common in video filenames
3. **ML Classifier** - Handle ambiguous/slang terms
4. **Metadata Analysis** - File size, extension patterns

### Example Python Code

```python
from better_profanity import profanity

def check_filename(filename):
    # Fast keyword check
    if profanity.contains_profanity(filename):
        return True

    # Add custom patterns
    porn_keywords = ['xxx', 'porn', 'adult', 'sex', 'nsfw']
    if any(kw in filename.lower() for kw in porn_keywords):
        return True

    return False
```

---

## Tool Comparison Matrix

| Tool | Size | Speed | Accuracy | Complexity |
|------|------|-------|----------|------------|
| better-profanity | <1 MB | Fastest | Low | Simple |
| profanity-check | ~5-10 MB | Fast | Medium | Simple |
| DistilBERT NSFW | ~268 MB | Slow | High | Medium |
| Custom ML | Variable | Variable | Variable | Complex |

---

## Platform Considerations

### For Android Apps
- **DistilBERT (268 MB)** - Too large for most mobile apps
- Consider **server-side API** for heavy ML models
- Use **lightweight keyword approach** on device

### For Server/Backend
- All options viable
- DistilBERT recommended for highest accuracy
- Can batch process for efficiency

---

## Sources

- [DistilBERT NSFW Text Classifier](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier)
- [Learning Strategies for Sensitive Content Detection](https://www.mdpi.com/2079-9292/12/11/2496)
- [profanity-check PyPI](https://pypi.org/project/profanity-check/)
- [better-profanity GitHub](https://github.com/snguyenthanh/better_profanity)
- [profanity-filter PyPI](https://pypi.org/project/profanity-filter/)
- [Replicate NSFW Detection](https://replicate.com/collections/detect-nsfw-content)
- [NSFW Text Identification Research](https://www.researchgate.net/publication/364652449_NSFW_Text_Identification)
- [Building a Better Profanity Detection Library](https://medium.com/data-science/building-a-better-profanity-detection-library-with-scikit-learn-3638b2f2c4c2)
