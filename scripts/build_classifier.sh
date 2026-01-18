#!/bin/bash
# Build the embedded classifier binary from trained model
# Usage: ./scripts/build_classifier.sh [model_dir]
#
# Default model_dir: training/bert-classifier-medium/final

set -e

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODEL_DIR="${1:-training/bert-classifier-medium/final}"
ONNX_DIR="training/bert-classifier-medium/onnx"

echo "=== Building Embedded Classifier ==="
echo "Model: $MODEL_DIR"
echo ""

# Check model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Create ONNX directory
mkdir -p "$ONNX_DIR"

# Step 1: Convert to ONNX and embed weights
echo "[1/3] Converting model to ONNX..."
python3 -c "
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import onnx
import shutil

MODEL_DIR = Path('$MODEL_DIR')
ONNX_DIR = Path('$ONNX_DIR')

print(f'  Loading model from {MODEL_DIR}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Create dummy input
dummy_text = 'test input'
inputs = tokenizer(dummy_text, return_tensors='pt', truncation=True, max_length=128)

# Export to ONNX (temporary file with external data)
temp_path = ONNX_DIR / 'model_temp.onnx'
print(f'  Exporting to ONNX...')

torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    temp_path,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch'},
    },
    opset_version=14,
)

# Load and save with embedded weights
print('  Embedding weights into single file...')
onnx_model = onnx.load(str(temp_path), load_external_data=True)
onnx.save_model(onnx_model, str(ONNX_DIR / 'model_embedded.onnx'), save_as_external_data=False)

# Clean up temp files
temp_path.unlink(missing_ok=True)
for f in ONNX_DIR.glob('model_temp*'):
    f.unlink()

# Copy tokenizer
print('  Copying tokenizer...')
shutil.copy(MODEL_DIR / 'tokenizer.json', ONNX_DIR / 'tokenizer.json')

# Report sizes
model_size = (ONNX_DIR / 'model_embedded.onnx').stat().st_size / 1024 / 1024
tok_size = (ONNX_DIR / 'tokenizer.json').stat().st_size / 1024
print(f'  Model: {model_size:.1f} MB')
print(f'  Tokenizer: {tok_size:.0f} KB')
"

# Step 2: Build Rust binary
echo ""
echo "[2/3] Building Rust binary..."
cargo build --release 2>&1 | tail -3

# Step 3: Report results
echo ""
echo "[3/3] Done!"
BINARY_SIZE=$(ls -lh target/release/mimmo_classifier | awk '{print $5}')
echo ""
echo "Binary: target/release/mimmo_classifier ($BINARY_SIZE)"
echo ""

# Quick test
echo "Quick test:"
./target/release/mimmo_classifier 2>&1 | head -8
