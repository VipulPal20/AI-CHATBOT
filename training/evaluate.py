import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from app.utils.config import settings

def evaluate_model():
    print("=== Starting Model Evaluation Pipeline ===")
    output_dir = settings.model.model_dir
    max_length = settings.model.max_length

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Model directory '{output_dir}' does not exist. Run training first.")

    # 1. Load Label Mappings
    with open(os.path.join(output_dir, "label_encoder.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)

    labels_list = [label_map[str(i)] for i in range(len(label_map))]

    # 2. Load Test Data
    test_df = pd.read_csv("data/processed/test.csv")
    texts = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    # 3. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. Perform Inference on Test Split
    y_pred = []
    probabilities = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

            y_pred.append(pred_class)
            probabilities.append(probs.tolist())

    # 5. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    class_report = classification_report(y_true, y_pred, target_names=labels_list, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "num_test_samples": len(texts),
        "confusion_matrix": conf_matrix,
        "per_class_metrics": class_report
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ Evaluation Results ================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Detailed metrics saved to: {metrics_path}")
    print("====================================================\n")

    return metrics

if __name__ == "__main__":
    evaluate_model()
