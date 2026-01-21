#!/usr/bin/env python3
"""
Fine-tune SmolLM-135M for torrent metadata extraction.

Usage:
    python scripts/train_smollm.py
    python scripts/train_smollm.py --resume_from_checkpoint models/smollm-metadata/checkpoint-11500

Requires: transformers, datasets, peft, accelerate, bitsandbytes
    pip install transformers datasets peft accelerate bitsandbytes
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Paths
TRAINING_DATA = Path(__file__).parent.parent / "data" / "llm_training_data.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "smollm-metadata"

# Model
MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"

# Training config
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 256


def load_training_data():
    """Load and format training data."""
    samples = []
    with open(TRAINING_DATA) as f:
        for line in f:
            data = json.loads(line)
            # Format as instruction-response for SmolLM
            # SmolLM-Instruct uses: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
            text = f"<|im_start|>user\n{data['input']}<|im_end|>\n<|im_start|>assistant\n{data['output']}<|im_end|>"
            samples.append({"text": text})
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM for metadata extraction")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME}")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Setup LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("Loading training data...")
    dataset = load_training_data()
    print(f"Loaded {len(dataset)} samples")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Split train/eval
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=device == "cuda",
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    else:
        print("Starting training from scratch...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Merge LoRA weights for inference
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_output = OUTPUT_DIR / "merged"
    merged_output.mkdir(exist_ok=True)
    merged_model.save_pretrained(merged_output)
    tokenizer.save_pretrained(merged_output)

    print(f"Done! Merged model saved to {merged_output}")
    print("\nTo test:")
    print(f"  python scripts/test_smollm.py")


if __name__ == "__main__":
    main()
