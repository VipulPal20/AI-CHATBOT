import os
import json
import re
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/intents.json"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/intent_classifier"

def clean_text(text: str) -> str:
    """Clean and normalize raw user text input."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def augment_patterns(tag: str, pattern: str) -> List[str]:
    """Generate subtle text variations for robust training data without leakage."""
    variations = [pattern]
    cleaned = clean_text(pattern)
    variations.append(cleaned)
    if cleaned.startswith("please "):
        variations.append(cleaned.replace("please ", ""))
    if " can you " in cleaned:
        variations.append(cleaned.replace(" can you ", " "))
    if " i want " in cleaned:
        variations.append(cleaned.replace(" i want ", " i need "))
    return list(set(variations))

def load_and_preprocess_dataset(
    raw_path: str = RAW_DATA_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, Dict]:
    """Load intents JSON, expand patterns, encode labels, and stratify split into train/val/test."""
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    records = []
    responses_map = {}

    for intent in raw_json.get("intents", []):
        tag = intent["tag"]
        responses_map[tag] = intent["responses"]
        for p in intent["patterns"]:
            augmented = augment_patterns(tag, p)
            for text_sample in augmented:
                records.append({"text": clean_text(text_sample), "intent": tag})

    df = pd.DataFrame(records).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])

    # Ensure output directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Stratified Split: Train, Validation, Test
    train_df, test_df = train_test_split(
        df, test_size=test_size + val_size, random_state=seed, stratify=df["label"]
    )
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        test_df, test_size=1.0 - relative_val_size, random_state=seed, stratify=test_df["label"]
    )

    # Save split datasets
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    # Save Label Mapping and Responses Map
    label_map = {int(idx): str(label) for idx, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(MODELS_DIR, "label_encoder.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    with open(os.path.join(MODELS_DIR, "responses.json"), "w", encoding="utf-8") as f:
        json.dump(responses_map, f, indent=2)

    print(f"Dataset Preprocessed Successfully:")
    print(f" - Total Samples: {len(df)}")
    print(f" - Train Samples: {len(train_df)}")
    print(f" - Validation Samples: {len(val_df)}")
    print(f" - Test Samples: {len(test_df)}")
    print(f" - Total Intent Classes: {len(label_encoder.classes_)}")

    return train_df, val_df, test_df, label_encoder, responses_map

if __name__ == "__main__":
    load_and_preprocess_dataset()
