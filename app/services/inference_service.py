import time
import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple

from app.models.model_loader import model_loader
from training.preprocess import clean_text

logger = logging.getLogger(__name__)

class InferenceService:
    """Service handling text preprocessing, model inference, and confidence calculation."""
    def __init__(self, use_onnx: bool = False):
        self.use_onnx = use_onnx
        model_loader.initialize(use_onnx=use_onnx)

    def predict(self, text: str) -> Dict[str, Any]:
        """Perform intent classification on input text string."""
        start_time = time.perf_counter()
        cleaned = clean_text(text)

        if not cleaned:
            return {
                "intent": "unknown_fallback",
                "confidence": 0.0,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "probabilities": {}
            }

        if model_loader.use_onnx and model_loader.onnx_session is not None:
            intent, confidence, probs_dict = self._predict_onnx(cleaned)
        else:
            intent, confidence, probs_dict = self._predict_pytorch(cleaned)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "latency_ms": latency_ms,
            "probabilities": probs_dict
        }

    def _predict_pytorch(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        encoding = model_loader.tokenizer.encode(text)
        probs_tensor = model_loader.pytorch_model.predict(
            encoding["input_ids"],
            encoding["attention_mask"]
        )[0]

        probs = probs_tensor.cpu().numpy()
        pred_class_idx = int(np.argmax(probs))
        confidence = float(probs[pred_class_idx])

        intent = model_loader.label_map.get(pred_class_idx, "unknown_fallback")
        probs_dict = {
            model_loader.label_map.get(i, f"class_{i}"): round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        return intent, confidence, probs_dict

    def _predict_onnx(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        encoded = model_loader.tokenizer.encode_numpy(text)
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
        logits = model_loader.onnx_session.run(None, inputs)[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        pred_class_idx = int(np.argmax(probs))
        confidence = float(probs[pred_class_idx])

        intent = model_loader.label_map.get(pred_class_idx, "unknown_fallback")
        probs_dict = {
            model_loader.label_map.get(i, f"class_{i}"): round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        return intent, confidence, probs_dict

inference_service = InferenceService()
