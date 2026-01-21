#!/usr/bin/env python3
"""
Test the fine-tuned SmolLM metadata extraction model.

Usage:
    python scripts/test_smollm.py [model_path]
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "smollm-metadata" / "merged"

TEST_CASES = [
    "<|extract|>[video/movie] The.Matrix.1999.1080p.BluRay.x264",
    "<|extract|>[video/movie] Inception.2010.REMASTERED.BluRay.1080p",
    "<|extract|>[audio/album] Nirvana (1992) Incesticide [FLAC]",
    "<|extract|>[audio/album] Pink Floyd - The Dark Side of the Moon (1973) [24-96]",
    "<|extract|>[video/episode] Breaking.Bad.S01E01.Pilot.720p.BluRay",
    "<|extract|>[video/season] Game.of.Thrones.S03.Complete.1080p.BluRay",
    "<|extract|>[audio/album] Taylor_Swift-folklore-2020-FLAC",
    "<|extract|>[video/movie] パルプ・フィクション.Pulp.Fiction.1994.BluRay.1080p",
]


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_MODEL)
    print(f"Loading model from: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    print("\n" + "=" * 60)
    print("Testing metadata extraction")
    print("=" * 60)

    for test_input in TEST_CASES:
        # Format as chat
        prompt = f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract just the assistant response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
            response = response.split("<|im_end|>")[0].strip()

        print(f"\nInput:  {test_input}")
        print(f"Output: {response}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
