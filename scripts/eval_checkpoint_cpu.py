#!/usr/bin/env python3
"""Evaluate a LoRA checkpoint on CPU with truncation logic."""
import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM-360M-Instruct"
TRAINING_DATA = Path(__file__).parent.parent / "data" / "llm_training_data.jsonl"

FIELD_COUNTS = {
    "video/movie": 2,
    "video/episode": 1,
    "video/season": 1,
    "video/series": 1,
    "audio/album": 3,
    "audio/track": 3,
}


def truncate_output(output: str, subtype: str, input_str: str) -> str:
    """Truncate output to expected field count and validate years."""
    max_fields = FIELD_COUNTS.get(subtype, 3)
    fields = [f.strip() for f in output.split("|")][:max_fields]

    # Validate years - if a field looks like a year, check it exists in input
    validated = []
    for field in fields:
        if re.match(r'^\d{4}$', field.strip()):
            if field.strip() in input_str:
                validated.append(field)
            # else skip hallucinated year
        else:
            validated.append(field)

    return " | ".join(validated)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "/tmp/checkpoint-2500"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32  # CPU needs float32
    )

    print(f"Loading LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()

    # Load test data
    print("Loading test data...")
    all_samples = []
    with open(TRAINING_DATA) as f:
        for line in f:
            all_samples.append(json.loads(line))

    # Use last 5% as test, sample from it
    test_start = int(len(all_samples) * 0.95)
    test_samples = all_samples[test_start:]

    import random
    random.seed(42)
    test_samples = random.sample(test_samples, min(num_samples, len(test_samples)))
    print(f"Evaluating on {len(test_samples)} samples (CPU - be patient)")

    exact_match = 0
    fuzzy_match = 0
    by_type = {}

    for i, sample in enumerate(test_samples):
        prompt = f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
            response = response.split("<|im_end>")[0].strip()

        subtype = sample.get("subtype", "")
        truncated = truncate_output(response, subtype, sample["input"])
        expected = sample["output"]

        # Score
        is_exact = truncated.strip().lower() == expected.strip().lower()
        sim = similarity(truncated, expected)
        is_fuzzy = sim >= 0.9

        if is_exact:
            exact_match += 1
        if is_fuzzy:
            fuzzy_match += 1

        # By type stats
        if subtype not in by_type:
            by_type[subtype] = {"exact": 0, "fuzzy": 0, "total": 0}
        by_type[subtype]["total"] += 1
        if is_exact:
            by_type[subtype]["exact"] += 1
        if is_fuzzy:
            by_type[subtype]["fuzzy"] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_samples)} - exact: {exact_match}/{i+1} ({exact_match/(i+1)*100:.1f}%)")

    # Results
    print("\n" + "="*60)
    print(f"RESULTS ({len(test_samples)} samples)")
    print("="*60)
    print(f"Exact match: {exact_match}/{len(test_samples)} ({exact_match/len(test_samples)*100:.1f}%)")
    print(f"Fuzzy match (>=90%): {fuzzy_match}/{len(test_samples)} ({fuzzy_match/len(test_samples)*100:.1f}%)")

    print("\nBy type:")
    for subtype, stats in sorted(by_type.items()):
        total = stats["total"]
        exact_pct = stats["exact"] / total * 100 if total else 0
        fuzzy_pct = stats["fuzzy"] / total * 100 if total else 0
        print(f"  {subtype}: {stats['exact']}/{total} exact ({exact_pct:.1f}%), {stats['fuzzy']}/{total} fuzzy ({fuzzy_pct:.1f}%)")


if __name__ == "__main__":
    main()
