import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from app.utils.config import settings

class IntentDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_model():
    print("=== Starting Model Training Pipeline ===")
    model_name = settings.model.name
    max_length = settings.model.max_length
    batch_size = settings.model.batch_size
    epochs = settings.model.epochs
    lr = settings.model.learning_rate
    output_dir = settings.model.model_dir

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Preprocessed Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    with open(os.path.join(output_dir, "label_encoder.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = len(label_map)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    train_dataset = IntentDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    val_dataset = IntentDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using compute device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={int(k): v for k, v in label_map.items()},
        label2id={v: int(k) for k, v in label_map.items()}
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        acc = accuracy_score(val_targets, val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="weighted", zero_division=0)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

        if f1 >= best_val_f1:
            best_val_f1 = f1
            model.save_pretrained(output_dir)
            print(f" Saved best model checkpoint to {output_dir}")

    print(f"=== Model Training Completed. Best Validation F1: {best_val_f1:.4f} ===")

if __name__ == "__main__":
    train_model()
