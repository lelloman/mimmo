#!/usr/bin/env python3
"""
Evaluate the cascade classifier against ground-truth training data.

Usage:
    python scripts/evaluate_classifier.py --samples 150
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass

import requests


OLLAMA_HOST = "http://192.168.1.92:11434"
MODEL = "qwen3:14b"


@dataclass
class Sample:
    content_type: str  # video/movie, video/series, audio/album
    input_text: str
    expected: dict


def normalize(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    return s.lower().strip()


def title_match(predicted: str, expected: str, threshold: float = 0.8) -> bool:
    """Check if titles match with fuzzy matching."""
    p = normalize(predicted)
    e = normalize(expected)

    if not p or not e:
        return False

    # Exact match
    if p == e:
        return True

    # Substring match
    if p in e or e in p:
        return True

    # Fuzzy match
    ratio = SequenceMatcher(None, p, e).ratio()
    return ratio >= threshold


def get_schema(content_type: str) -> dict:
    """Get JSON schema for content type."""
    if content_type == "audio/album":
        return {
            "artist": "string or null",
            "album_name": "string or null",
            "year": "number or null"
        }
    elif content_type == "video/movie":
        return {"title": "string or null", "year": "number or null"}
    elif content_type == "video/series":
        return {"series_title": "string or null"}
    return {}


def build_prompt(input_text: str, content_type: str) -> str:
    """Build the extraction prompt."""
    schema = get_schema(content_type)
    schema_str = json.dumps(schema, indent=2)

    return f"""Extract metadata from this torrent listing. The content type is: {content_type}

Input:
{input_text}

Extract ONLY the following fields as JSON:
{schema_str}

Rules:
- Extract the actual content title/name, NOT technical metadata like resolution, codec, release group
- For series, extract ONLY the series name (e.g., "Breaking Bad"), not episode codes like S01E01
- For audio, distinguish between artist name and album name
- Use null for fields you cannot determine
- Return ONLY valid JSON, no explanation

JSON:"""


def query_ollama(prompt: str) -> str:
    """Query Ollama."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512}
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        return ""

    return response.json().get("response", "")


def parse_json_response(response: str) -> dict | None:
    """Parse JSON from response."""
    # Remove thinking blocks
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response
    match = re.search(r'\{[^{}]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def evaluate_sample(sample: Sample) -> dict:
    """Evaluate a single sample."""
    prompt = build_prompt(sample.input_text, sample.content_type)
    response = query_ollama(prompt)
    predicted = parse_json_response(response)

    if predicted is None:
        return {
            "match": False,
            "error": "parse_failed",
            "expected": sample.expected,
            "predicted": None,
            "raw_response": response[:200]
        }

    # Compare based on type
    if sample.content_type == "video/movie":
        title_ok = title_match(predicted.get("title", ""), sample.expected.get("title", ""))
        match = title_ok

    elif sample.content_type == "video/series":
        title_ok = title_match(predicted.get("series_title", ""), sample.expected.get("series_title", ""))
        match = title_ok

    elif sample.content_type == "audio/album":
        album_ok = title_match(predicted.get("album_name", ""), sample.expected.get("album_name", ""))
        artist_ok = title_match(predicted.get("artist", ""), sample.expected.get("artist", "")) if sample.expected.get("artist") else True
        match = album_ok and artist_ok

    else:
        match = False

    return {
        "match": match,
        "expected": sample.expected,
        "predicted": predicted,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier against ground truth")
    parser.add_argument("--samples", type=int, default=150, help="Number of samples to test")
    parser.add_argument("--data", type=str, default="data/training_samples.jsonl", help="Training data file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all results")
    args = parser.parse_args()

    # Load all samples
    print(f"Loading samples from {args.data}...")
    all_samples = []
    with open(args.data) as f:
        for line in f:
            data = json.loads(line)
            all_samples.append(Sample(
                content_type=data["label"]["type"],
                input_text=data["input"],
                expected=data["label"]
            ))

    # Group by type
    by_type = {}
    for s in all_samples:
        if s.content_type not in by_type:
            by_type[s.content_type] = []
        by_type[s.content_type].append(s)

    print(f"Total samples: {len(all_samples)}")
    for t, samples in by_type.items():
        print(f"  {t}: {len(samples)}")

    # Sample evenly from each type
    samples_per_type = args.samples // len(by_type)
    test_samples = []
    for t, samples in by_type.items():
        random.shuffle(samples)
        test_samples.extend(samples[:samples_per_type])

    random.shuffle(test_samples)
    print(f"\nTesting {len(test_samples)} samples ({samples_per_type} per type)...")
    print(f"Model: {MODEL} @ {OLLAMA_HOST}")
    print("-" * 60)

    # Evaluate
    results = {"video/movie": [], "video/series": [], "audio/album": []}
    start_time = time.time()

    for i, sample in enumerate(test_samples):
        result = evaluate_sample(sample)
        results[sample.content_type].append(result)

        status = "✓" if result["match"] else "✗"
        if args.verbose or not result["match"]:
            print(f"\n[{i+1}] {status} {sample.content_type}")
            print(f"  Input: {sample.input_text[:70]}...")
            print(f"  Expected: {result['expected']}")
            print(f"  Predicted: {result['predicted']}")

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"\n--- Progress: {i+1}/{len(test_samples)} ({rate:.1f} samples/sec) ---")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_correct = 0
    total_count = 0

    for t in ["video/movie", "video/series", "audio/album"]:
        type_results = results.get(t, [])
        if not type_results:
            continue
        correct = sum(1 for r in type_results if r["match"])
        total = len(type_results)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"{t}: {correct}/{total} ({accuracy:.1f}%)")
        total_correct += correct
        total_count += total

    overall = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\nOVERALL: {total_correct}/{total_count} ({overall:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({total_count/elapsed:.1f} samples/sec)")


if __name__ == "__main__":
    main()
