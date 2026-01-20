#!/usr/bin/env python3
"""
Cascade-based metadata extraction using multiple LLMs.

Strategy:
1. Run qwen3 (GPU) + gemma-4b (halos) in parallel - fast
2. If they agree -> high confidence
3. If they disagree -> run gpt-oss-120b as tiebreaker

Usage:
    python scripts/cascade_extraction.py --limit 20 --no-force-json
"""

import argparse
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import requests


@dataclass
class Sample:
    """A torrent sample with metadata."""
    number: int
    name: str
    total_size: str
    files: list[tuple[str, str]]
    medium: str
    subcategory: str | None
    confidence: float


def parse_evaluation_md(filepath: Path) -> list[Sample]:
    """Parse samples from extraction_evaluation.md."""
    content = filepath.read_text()
    samples = []

    sample_blocks = re.split(r'### Sample (\d+)', content)[1:]

    for i in range(0, len(sample_blocks), 2):
        number = int(sample_blocks[i])
        block = sample_blocks[i + 1]

        name_match = re.search(r'\*\*Torrent Name:\*\* `([^`]+)`', block)
        if not name_match:
            continue
        name = name_match.group(1)

        size_match = re.search(r'\*\*Total Size:\*\* ([^\n]+)', block)
        total_size = size_match.group(1) if size_match else "unknown"

        files = []
        files_section = re.search(r'\*\*Top Files:\*\*\n(.*?)(?=\n\n|\*\*Mimmo)', block, re.DOTALL)
        if files_section:
            for line in files_section.group(1).strip().split('\n'):
                file_match = re.match(r'- `([^`]+)` \(([^)]+)\)', line)
                if file_match:
                    files.append((file_match.group(1), file_match.group(2)))

        class_match = re.search(r'\*\*Mimmo Classification:\*\* `([^`]+)`(?: / `([^`]+)`)? \(conf: ([0-9.]+)\)', block)
        if class_match:
            medium = class_match.group(1)
            subcategory = class_match.group(2)
            confidence = float(class_match.group(3))
        else:
            medium = "unknown"
            subcategory = None
            confidence = 0.0

        samples.append(Sample(
            number=number,
            name=name,
            total_size=total_size,
            files=files,
            medium=medium,
            subcategory=subcategory,
            confidence=confidence,
        ))

    return samples


def format_input(sample: Sample) -> str:
    """Format sample for LLM input."""
    lines = [sample.name]
    for filename, size in sample.files[:3]:
        lines.append(f"{filename} ({size})")
    return "\n".join(lines)


def get_schema_for_sample(sample: Sample) -> dict:
    """Get the JSON schema based on medium/subcategory."""
    if sample.medium == "audio":
        return {
            "artist": "string or null",
            "album": "string or null",
            "track_name": "string or null",
            "year": "number or null"
        }
    elif sample.medium == "video":
        if sample.subcategory == "movie":
            return {"title": "string or null", "year": "number or null"}
        else:
            return {"series_title": "string or null"}
    return {}


def build_prompt(sample: Sample, input_text: str) -> str:
    """Build the extraction prompt."""
    schema = get_schema_for_sample(sample)
    schema_str = json.dumps(schema, indent=2)

    medium_desc = sample.medium
    if sample.subcategory:
        medium_desc = f"{sample.medium}/{sample.subcategory}"

    return f"""Extract metadata from this torrent listing. The content type is: {medium_desc}

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


def query_ollama(prompt: str, model: str, host: str) -> tuple[str, float]:
    """Query Ollama API."""
    start = time.time()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512}
    }

    response = requests.post(
        f"{host}/api/generate",
        json=payload,
        timeout=120
    )
    elapsed = time.time() - start

    if response.status_code != 200:
        return f"ERROR: {response.status_code}", elapsed

    return response.json().get("response", ""), elapsed


def query_llamacpp(prompt: str, host: str, model: str) -> tuple[str, float]:
    """Query llama.cpp server (chat API)."""
    start = time.time()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    response = requests.post(
        f"{host}/v1/chat/completions",
        json=payload,
        timeout=120
    )
    elapsed = time.time() - start

    if response.status_code != 200:
        return f"ERROR: {response.status_code}", elapsed

    choices = response.json().get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return msg.get("content", ""), elapsed
    return "", elapsed


def parse_json_response(response: str) -> dict | None:
    """Parse JSON from response, handling thinking blocks and markdown."""
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    json_block = re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```', cleaned, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def normalize_value(value) -> str:
    """Normalize a value for comparison."""
    if value is None:
        return ""
    s = str(value).lower().strip()
    s = re.sub(r'[._]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def compare_extractions(parsed1: dict | None, parsed2: dict | None, sample: Sample) -> dict:
    """Compare two extractions and return comparison result."""
    if parsed1 is None and parsed2 is None:
        return {"status": "both_null", "agreement": 1.0, "details": {}}

    if parsed1 is None:
        return {"status": "model1_failed", "agreement": 0.0, "details": {}}

    if parsed2 is None:
        return {"status": "model2_failed", "agreement": 0.0, "details": {}}

    if sample.medium == "audio":
        fields = ["artist", "album", "track_name", "year"]
    elif sample.medium == "video":
        if sample.subcategory == "movie":
            fields = ["title", "year"]
        else:
            fields = ["series_title"]
    else:
        fields = []

    details = {}
    agreements = []
    exact_match_fields = {"year"}

    for field in fields:
        v1 = parsed1.get(field)
        v2 = parsed2.get(field)

        n1 = normalize_value(v1)
        n2 = normalize_value(v2)

        if n1 == n2:
            similarity = 1.0
        elif not n1 and not n2:
            similarity = 1.0
        elif not n1 or not n2:
            similarity = 0.0
        elif field in exact_match_fields:
            similarity = 0.0
        else:
            similarity = SequenceMatcher(None, n1, n2).ratio()

        details[field] = {
            "model1": v1,
            "model2": v2,
            "similarity": similarity,
            "match": similarity >= 0.7
        }
        agreements.append(similarity)

    avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0

    if avg_agreement >= 0.7:
        status = "agree"
    else:
        status = "disagree"

    return {
        "status": status,
        "agreement": avg_agreement,
        "details": details
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-model", default="qwen3-coder:30b")
    parser.add_argument("--ollama-host", default="http://192.168.1.92:11434")
    parser.add_argument("--ollama-workers", type=int, default=2)
    parser.add_argument("--llamacpp-host", action="append", default=[])
    parser.add_argument("--fast-model", default="gemma-3-4b-it-Q4_K_M")
    parser.add_argument("--tiebreaker-model", default="gpt-oss-120b-Q4_K_M")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-force-json", action="store_true")
    args = parser.parse_args()

    if not args.llamacpp_host:
        args.llamacpp_host = ["http://192.168.1.102:8080", "http://192.168.1.103:8080"]

    # Load samples
    eval_path = Path(__file__).parent.parent / "data" / "extraction_evaluation.md"
    samples = parse_evaluation_md(eval_path)
    av_samples = [s for s in samples if s.medium in ("audio", "video")]

    if args.limit:
        av_samples = av_samples[:args.limit]

    print(f"Loaded {len(av_samples)} audio/video samples")
    print(f"Primary: {args.ollama_model} (GPU) + {args.fast_model} (halos)")
    print(f"Tiebreaker: {args.tiebreaker_model}")
    print("-" * 60)

    # Check connections
    try:
        requests.get(f"{args.ollama_host}/api/tags", timeout=5)
        print(f"✓ Ollama connected")
    except:
        print(f"✗ Ollama not available")
        return

    for host in args.llamacpp_host:
        try:
            resp = requests.get(f"{host}/v1/models", timeout=5)
            models = [m["id"] for m in resp.json().get("data", [])]
            print(f"✓ llama.cpp @ {host}: {len(models)} models")
        except Exception as e:
            print(f"✗ llama.cpp @ {host}: {e}")

    print("-" * 60)

    start_time = time.time()

    # PHASE 1: Run qwen + gemma in parallel
    print("Phase 1: Running qwen + gemma...")

    qwen_results = {}
    gemma_results = {}

    total_workers = args.ollama_workers + len(args.llamacpp_host)

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {}

        # Submit qwen tasks
        for sample in av_samples:
            input_text = format_input(sample)
            prompt = build_prompt(sample, input_text)
            future = executor.submit(query_ollama, prompt, args.ollama_model, args.ollama_host)
            futures[future] = ("qwen", sample, input_text, prompt)

        # Submit gemma tasks (round-robin across hosts)
        for i, sample in enumerate(av_samples):
            host = args.llamacpp_host[i % len(args.llamacpp_host)]
            input_text = format_input(sample)
            prompt = build_prompt(sample, input_text)
            future = executor.submit(query_llamacpp, prompt, host, args.fast_model)
            futures[future] = ("gemma", sample, input_text, prompt)

        # Collect results
        qwen_done = 0
        gemma_done = 0
        for future in as_completed(futures):
            model_type, sample, input_text, prompt = futures[future]
            response, elapsed = future.result()
            parsed = parse_json_response(response)

            if model_type == "qwen":
                qwen_results[sample.number] = {
                    "response": response, "parsed": parsed, "time": elapsed,
                    "input": input_text, "prompt": prompt
                }
                qwen_done += 1
                status = "✓" if parsed else "✗"
                print(f"[qwen {qwen_done}/{len(av_samples)}] Sample {sample.number}: {status} ({elapsed:.1f}s)")
            else:
                gemma_results[sample.number] = {
                    "response": response, "parsed": parsed, "time": elapsed
                }
                gemma_done += 1
                status = "✓" if parsed else "✗"
                print(f"[gemma {gemma_done}/{len(av_samples)}] Sample {sample.number}: {status} ({elapsed:.1f}s)")

    phase1_time = time.time() - start_time

    # PHASE 2: Compare and identify disagreements
    print("-" * 60)
    print("Phase 2: Comparing results...")

    agreed = []
    disagreed = []

    for sample in av_samples:
        qwen_res = qwen_results.get(sample.number, {})
        gemma_res = gemma_results.get(sample.number, {})

        comparison = compare_extractions(qwen_res.get("parsed"), gemma_res.get("parsed"), sample)

        if comparison["status"] == "agree" or comparison["status"] == "both_null":
            agreed.append({
                "sample": sample,
                "qwen": qwen_res,
                "gemma": gemma_res,
                "comparison": comparison
            })
            print(f"Sample {sample.number}: ✓ agree ({comparison['agreement']:.0%})")
        else:
            disagreed.append({
                "sample": sample,
                "qwen": qwen_res,
                "gemma": gemma_res,
                "comparison": comparison
            })
            print(f"Sample {sample.number}: ✗ disagree ({comparison['agreement']:.0%})")

    print(f"\nPhase 1 complete: {len(agreed)} agree, {len(disagreed)} disagree")

    # PHASE 3: Run tiebreaker on disagreements
    tiebreaker_results = {}
    if disagreed:
        print("-" * 60)
        print(f"Phase 3: Running tiebreaker ({args.tiebreaker_model}) on {len(disagreed)} samples...")

        with ThreadPoolExecutor(max_workers=len(args.llamacpp_host)) as executor:
            futures = {}
            for i, item in enumerate(disagreed):
                sample = item["sample"]
                host = args.llamacpp_host[i % len(args.llamacpp_host)]
                prompt = item["qwen"]["prompt"]
                future = executor.submit(query_llamacpp, prompt, host, args.tiebreaker_model)
                futures[future] = (sample, item)

            for future in as_completed(futures):
                sample, item = futures[future]
                response, elapsed = future.result()
                parsed = parse_json_response(response)
                tiebreaker_results[sample.number] = {
                    "response": response, "parsed": parsed, "time": elapsed
                }
                status = "✓" if parsed else "✗"
                print(f"[tiebreaker] Sample {sample.number}: {status} ({elapsed:.1f}s)")

    total_time = time.time() - start_time

    # PHASE 4: Final determination
    print("-" * 60)
    print("Phase 4: Final results...")

    final_results = []

    # Agreed samples -> use qwen's result (or either, they match)
    for item in agreed:
        final_results.append({
            "sample": item["sample"],
            "status": "agreed",
            "confidence": "high",
            "result": item["qwen"]["parsed"],
            "source": "qwen+gemma"
        })

    # Disagreed samples -> check tiebreaker
    for item in disagreed:
        sample = item["sample"]
        qwen_parsed = item["qwen"].get("parsed")
        gemma_parsed = item["gemma"].get("parsed")
        tie_parsed = tiebreaker_results.get(sample.number, {}).get("parsed")

        # Check if tiebreaker agrees with qwen
        qwen_tie = compare_extractions(qwen_parsed, tie_parsed, sample)
        # Check if tiebreaker agrees with gemma
        gemma_tie = compare_extractions(gemma_parsed, tie_parsed, sample)

        if qwen_tie["agreement"] >= 0.7:
            final_results.append({
                "sample": sample,
                "status": "tiebreaker_qwen",
                "confidence": "medium",
                "result": qwen_parsed,
                "source": "qwen+gpt"
            })
            print(f"Sample {sample.number}: tiebreaker agrees with qwen")
        elif gemma_tie["agreement"] >= 0.7:
            final_results.append({
                "sample": sample,
                "status": "tiebreaker_gemma",
                "confidence": "medium",
                "result": gemma_parsed,
                "source": "gemma+gpt"
            })
            print(f"Sample {sample.number}: tiebreaker agrees with gemma")
        else:
            final_results.append({
                "sample": sample,
                "status": "no_consensus",
                "confidence": "low",
                "result": qwen_parsed,  # Default to qwen
                "source": "qwen (no consensus)"
            })
            print(f"Sample {sample.number}: no consensus")

    # Generate report
    output_path = Path(__file__).parent.parent / "data" / "cascade_extraction.md"

    high_conf = sum(1 for r in final_results if r["confidence"] == "high")
    med_conf = sum(1 for r in final_results if r["confidence"] == "medium")
    low_conf = sum(1 for r in final_results if r["confidence"] == "low")

    with open(output_path, "w") as f:
        f.write("# Cascade Extraction Results\n\n")
        f.write(f"**Primary models:** {args.ollama_model} + {args.fast_model}\n\n")
        f.write(f"**Tiebreaker:** {args.tiebreaker_model}\n\n")
        f.write(f"**Total samples:** {len(final_results)}\n\n")
        f.write(f"**Phase 1 time:** {phase1_time:.1f}s\n\n")
        f.write(f"**Total time:** {total_time:.1f}s\n\n")

        f.write("## Confidence Summary\n\n")
        f.write(f"- **High confidence (qwen+gemma agree):** {high_conf} ({100*high_conf/len(final_results):.1f}%)\n")
        f.write(f"- **Medium confidence (2/3 agree):** {med_conf} ({100*med_conf/len(final_results):.1f}%)\n")
        f.write(f"- **Low confidence (no consensus):** {low_conf} ({100*low_conf/len(final_results):.1f}%)\n\n")

        f.write("## High Confidence Results\n\n")
        f.write("| # | Name | Result | Source |\n")
        f.write("|---|------|--------|--------|\n")
        for r in final_results:
            if r["confidence"] == "high":
                s = r["sample"]
                name = s.name[:40] + "..." if len(s.name) > 40 else s.name
                result = str(r["result"])[:40] if r["result"] else "-"
                f.write(f"| {s.number} | {name} | {result} | {r['source']} |\n")

        f.write("\n## Medium Confidence Results\n\n")
        f.write("| # | Name | Result | Source |\n")
        f.write("|---|------|--------|--------|\n")
        for r in final_results:
            if r["confidence"] == "medium":
                s = r["sample"]
                name = s.name[:40] + "..." if len(s.name) > 40 else s.name
                result = str(r["result"])[:40] if r["result"] else "-"
                f.write(f"| {s.number} | {name} | {result} | {r['source']} |\n")

        f.write("\n## Low Confidence (Needs Review)\n\n")
        for r in final_results:
            if r["confidence"] == "low":
                s = r["sample"]
                f.write(f"### Sample {s.number}\n\n")
                f.write(f"**Input:** `{s.name[:60]}...`\n\n")
                f.write(f"**Result:** `{r['result']}`\n\n")

    print("-" * 60)
    print(f"Results written to {output_path}")
    print(f"\nSummary:")
    print(f"  High confidence: {high_conf}/{len(final_results)} ({100*high_conf/len(final_results):.1f}%)")
    print(f"  Medium confidence: {med_conf}/{len(final_results)} ({100*med_conf/len(final_results):.1f}%)")
    print(f"  Low confidence: {low_conf}/{len(final_results)} ({100*low_conf/len(final_results):.1f}%)")
    print(f"  Total time: {total_time:.1f}s (Phase 1: {phase1_time:.1f}s)")


if __name__ == "__main__":
    main()
