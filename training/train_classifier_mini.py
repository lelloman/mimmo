#!/usr/bin/env python3
"""Train BERT-mini classifier for torrent medium type classification (5-class)."""

import json
import sqlite3
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# Configuration
MODEL_NAME = "prajjwal1/bert-mini"  # 4 layers, 256 hidden, ~11M params
DATA_DIR = Path(__file__).parent.parent / "data"
CONSENSUS_DATA = DATA_DIR / "training_data_consensus.jsonl"
OUTPUT_DIR = Path(__file__).parent / "bert-classifier-mini"
NUM_LABELS = 5

LABEL2ID = {
    "audio": 0,
    "video": 1,
    "software": 2,
    "book": 3,
    "other": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_data(file_path):
    """Load JSONL training data."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            label = item["label"]
            if label not in LABEL2ID:
                continue
            data.append({
                "text": item["text"],
                "label": LABEL2ID[label],
            })
    return data


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main():
    print(f"Loading data from {CONSENSUS_DATA}...")
    data = load_data(CONSENSUS_DATA)
    print(f"Loaded {len(data)} samples")

    # Print label distribution
    from collections import Counter
    label_counts = Counter(d["label"] for d in data)
    print("\nLabel distribution:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  {ID2LABEL[label_id]}: {count}")

    # Create dataset and split
    dataset = Dataset.from_list(data)
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"\nTrain: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # slightly lower for larger model
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
    )

    # Compute class weights for imbalanced data
    total = sum(label_counts.values())
    class_weights = torch.tensor([
        total / (NUM_LABELS * label_counts[i]) for i in range(NUM_LABELS)
    ], dtype=torch.float32)

    if torch.cuda.is_available():
        class_weights = class_weights.cuda()

    print(f"\nClass weights: {class_weights.tolist()}")

    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Train
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Detailed classification report
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)]
    ))

    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print("\nDone!")


if __name__ == "__main__":
    main()
