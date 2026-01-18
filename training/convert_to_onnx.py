#!/usr/bin/env python3
"""Convert BERT-tiny classifier to ONNX format for Rust inference."""

from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shutil

MODEL_DIR = Path(__file__).parent / "bert-classifier-medium" / "final"
ONNX_DIR = Path(__file__).parent / "bert-classifier-medium" / "onnx"


def main():
    ONNX_DIR.mkdir(exist_ok=True)

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Create dummy input for export
    dummy_text = "test input"
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=128)

    # Export to ONNX
    onnx_path = ONNX_DIR / "model.onnx"
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

    # Copy tokenizer files needed for Rust
    print("Copying tokenizer files...")
    for fname in ["tokenizer.json", "vocab.txt", "config.json"]:
        src = MODEL_DIR / fname
        if src.exists():
            shutil.copy(src, ONNX_DIR / fname)

    # List output files
    print("\nOutput files:")
    for f in sorted(ONNX_DIR.iterdir()):
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

    test_input = "Ubuntu 22.04 LTS Desktop ISO\nubuntu-22.04-desktop-amd64.iso (3.6GB)"
    inputs = tokenizer(test_input, return_tensors="np", truncation=True, max_length=128)

    outputs = session.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )

    logits = outputs[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    pred_id = probs.argmax()

    id2label = {0: "audio", 1: "video", 2: "software", 3: "book", 4: "other"}
    print(f"  Input: {test_input[:50]}...")
    print(f"  Prediction: {id2label[pred_id]} ({probs[0][pred_id]:.1%})")

    print("\nDone!")


if __name__ == "__main__":
    main()
