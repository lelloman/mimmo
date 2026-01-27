#!/usr/bin/env python3
"""Convert BERT-tiny classifier to ONNX format for Rust inference.

Usage:
    python convert_to_onnx.py          # Convert medium classifier
    python convert_to_onnx.py --nsfw   # Convert NSFW classifier
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shutil


def main():
    parser = argparse.ArgumentParser(description="Convert BERT classifier to ONNX")
    parser.add_argument("--nsfw", action="store_true", help="Convert NSFW classifier")
    parser.add_argument("--model-dir", type=Path, help="Override model directory")
    args = parser.parse_args()

    if args.nsfw:
        # NSFW classifier
        model_dir = args.model_dir or Path("bert-classifier-nsfw") / "final"
        onnx_dir = Path("bert-classifier-nsfw") / "onnx"
        id2label = {0: "safe", 1: "nsfw"}
        test_input = "Tokyo.Hot.XXX.JAV.Uncensored"
    else:
        # Medium classifier
        model_dir = args.model_dir or Path(__file__).parent / "bert-classifier-medium" / "final"
        onnx_dir = Path(__file__).parent / "bert-classifier-medium" / "onnx"
        id2label = {0: "audio", 1: "video", 2: "software", 3: "book", 4: "other"}
        test_input = "Ubuntu 22.04 LTS Desktop ISO\nubuntu-22.04-desktop-amd64.iso (3.6GB)"

    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Create dummy input for export
    dummy_text = "test input"
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=128)

    # Export to ONNX
    onnx_path = onnx_dir / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )

    # Also create embedded version for Rust (merges external data into single file)
    print("Creating embedded version for Rust...")
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    model_onnx = onnx.load(str(onnx_path))
    # Load any external data that may have been created
    try:
        load_external_data_for_model(model_onnx, str(onnx_dir))
    except Exception:
        pass  # No external data, that's fine

    embedded_path = onnx_dir / "model_embedded.onnx"
    onnx.save_model(model_onnx, str(embedded_path), save_as_external_data=False)
    print(f"Saved embedded model to {embedded_path}")

    # Copy tokenizer files needed for Rust (only if not NSFW - NSFW shares tokenizer)
    if not args.nsfw:
        print("Copying tokenizer files...")
        for fname in ["tokenizer.json", "vocab.txt", "config.json"]:
            src = model_dir / fname
            if src.exists():
                shutil.copy(src, onnx_dir / fname)
    else:
        print("NSFW model - skipping tokenizer (shared with medium classifier)")

    # List output files
    print("\nOutput files:")
    for f in sorted(onnx_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")
        elif size > 1024:
            print(f"  {f.name}: {size / 1024:.1f} KB")
        else:
            print(f"  {f.name}: {size} B")

    # Test with onnxruntime
    print("\nTesting ONNX model with onnxruntime...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(str(onnx_path))

    inputs = tokenizer(test_input, return_tensors="np", truncation=True, max_length=128)

    outputs = session.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )

    logits = outputs[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    pred_id = probs.argmax()

    print(f"  Input: {test_input[:50]}...")
    print(f"  Prediction: {id2label[pred_id]} ({probs[0][pred_id]:.1%})")

    print("\nDone!")


if __name__ == "__main__":
    main()
