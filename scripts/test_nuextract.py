#!/usr/bin/env python3
"""Test NuExtract for entity extraction on torrent names."""

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test samples - mix of audio and video
AUDIO_SAMPLES = [
    "Pink Floyd - The Dark Side of the Moon (1973) [FLAC]",
    "Led Zeppelin - IV (1971) [24-96 Vinyl Rip]",
    "The Beatles - Abbey Road [320kbps MP3]",
    "Radiohead - OK Computer 1997 FLAC",
    "Miles Davis - Kind of Blue (1959) [DSD256]",
    "Various Artists - Now That's What I Call Music 100 (2018)",
    "Queen - Greatest Hits I II & III The Platinum Collection",
    "Nirvana - Nevermind (20th Anniversary Super Deluxe Edition) [FLAC]",
    "Bob Dylan Discography 1962-2020 [MP3 320]",
    "Daft Punk - Random Access Memories 2013 [24bit Hi-Res]",
]

VIDEO_SAMPLES = [
    "The Shawshank Redemption 1994 1080p BluRay x264",
    "Game of Thrones S01E01 Winter Is Coming 1080p",
    "Breaking Bad S05E16 Felina 720p BluRay x264",
    "Inception 2010 2160p UHD BluRay REMUX HDR HEVC",
    "The Office US S01-S09 Complete 720p WEB-DL",
    "Stranger Things S04 Complete 1080p NF WEB-DL",
    "Pulp Fiction 1994 REMASTERED 1080p BluRay x265",
    "Avatar The Way of Water 2022 2160p WEB-DL DDP5.1 Atmos",
    "The Mandalorian S03E08 Chapter 24 The Return 2160p DSNP WEB-DL",
    "Oppenheimer 2023 1080p WEBRip x264 AAC",
]

# JSON schemas for extraction
AUDIO_SCHEMA = json.dumps({
    "artist": "",
    "album": "",
    "year": "",
    "audio_format": "",
    "bitrate": ""
})

VIDEO_SCHEMA = json.dumps({
    "title": "",
    "year": "",
    "season": "",
    "episode": "",
    "resolution": "",
    "codec": ""
})


def create_prompt(text: str, schema: str) -> str:
    """Create NuExtract prompt format."""
    return f"""<|input|>
### Template:
{schema}

### Text:
{text}
<|output|>
"""


def extract(model, tokenizer, text: str, schema: str, device: str) -> dict:
    """Run extraction on a single sample."""
    prompt = create_prompt(text, schema)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Avoid cache compatibility issues
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Try to parse as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {"raw": response.strip()}


def extract_batch(model, tokenizer, texts: list[str], schema: str, device: str) -> list[dict]:
    """Run extraction on a batch of samples."""
    prompts = [create_prompt(text, schema) for text in texts]

    # Tokenize with padding
    tokenizer.padding_side = "left"  # For generation, pad on left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_lengths = [len(tokenizer.encode(p)) for p in prompts]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    results = []
    for i, output in enumerate(outputs):
        # Find where the padding ends and real content starts
        # Decode only the new tokens (after the input)
        response = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            results.append(json.loads(response.strip()))
        except json.JSONDecodeError:
            results.append({"raw": response.strip()})

    return results


def test_audio(model, tokenizer, device):
    print("=" * 60)
    print("AUDIO ENTITY EXTRACTION")
    print("=" * 60)

    total_time = 0
    for sample in AUDIO_SAMPLES:
        start = time.perf_counter()
        result = extract(model, tokenizer, sample, AUDIO_SCHEMA, device)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        print(f"\nInput: {sample}")
        print(f"Time: {elapsed*1000:.1f}ms")
        for key, value in result.items():
            if value:
                print(f"  {key:12} → {value!r}")

    print(f"\nAvg time: {total_time/len(AUDIO_SAMPLES)*1000:.1f}ms")


def test_video(model, tokenizer, device):
    print("\n" + "=" * 60)
    print("VIDEO ENTITY EXTRACTION")
    print("=" * 60)

    total_time = 0
    for sample in VIDEO_SAMPLES:
        start = time.perf_counter()
        result = extract(model, tokenizer, sample, VIDEO_SCHEMA, device)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        print(f"\nInput: {sample}")
        print(f"Time: {elapsed*1000:.1f}ms")
        for key, value in result.items():
            if value:
                print(f"  {key:12} → {value!r}")

    print(f"\nAvg time: {total_time/len(VIDEO_SAMPLES)*1000:.1f}ms")


def test_batched(model, tokenizer, device, batch_size: int):
    print("\n" + "=" * 60)
    print(f"BATCHED EXTRACTION (batch_size={batch_size})")
    print("=" * 60)

    # Test with audio samples
    print("\n--- Audio Batched ---")
    for i in range(0, len(AUDIO_SAMPLES), batch_size):
        batch = AUDIO_SAMPLES[i:i+batch_size]
        start = time.perf_counter()
        results = extract_batch(model, tokenizer, batch, AUDIO_SCHEMA, device)
        elapsed = time.perf_counter() - start

        print(f"\nBatch {i//batch_size + 1}: {len(batch)} samples in {elapsed*1000:.1f}ms ({elapsed/len(batch)*1000:.1f}ms/sample)")
        for sample, result in zip(batch, results):
            print(f"  {sample[:50]}...")
            for key, value in result.items():
                if value:
                    print(f"    {key:12} → {value!r}")

    # Test with video samples
    print("\n--- Video Batched ---")
    for i in range(0, len(VIDEO_SAMPLES), batch_size):
        batch = VIDEO_SAMPLES[i:i+batch_size]
        start = time.perf_counter()
        results = extract_batch(model, tokenizer, batch, VIDEO_SCHEMA, device)
        elapsed = time.perf_counter() - start

        print(f"\nBatch {i//batch_size + 1}: {len(batch)} samples in {elapsed*1000:.1f}ms ({elapsed/len(batch)*1000:.1f}ms/sample)")
        for sample, result in zip(batch, results):
            print(f"  {sample[:50]}...")
            for key, value in result.items():
                if value:
                    print(f"    {key:12} → {value!r}")

    # Combined benchmark
    print("\n--- Combined Benchmark ---")
    all_samples = AUDIO_SAMPLES + VIDEO_SAMPLES
    # Use same schema for simplicity in benchmark
    combined_schema = json.dumps({
        "title": "",
        "artist": "",
        "album": "",
        "year": "",
        "season": "",
        "episode": "",
        "resolution": "",
        "format": ""
    })

    start = time.perf_counter()
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i:i+batch_size]
        results = extract_batch(model, tokenizer, batch, combined_schema, device)
    total_time = time.perf_counter() - start

    print(f"Total: {len(all_samples)} samples in {total_time*1000:.1f}ms")
    print(f"Throughput: {len(all_samples)/total_time:.2f} samples/sec ({total_time/len(all_samples)*1000:.1f}ms/sample)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="numind/NuExtract",
                        help="Model to use (numind/NuExtract, numind/NuExtract-tiny, numind/NuExtract-large)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size for batched extraction (0 = sequential only)")
    parser.add_argument("--batch-only", action="store_true",
                        help="Only run batched tests, skip sequential")
    args = parser.parse_args()

    print(f"Loading {args.model}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if not args.batch_only:
        test_audio(model, tokenizer, device)
        test_video(model, tokenizer, device)

    if args.batch_size > 0:
        test_batched(model, tokenizer, device, args.batch_size)
