import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from app.utils.config import settings

def export_to_onnx():
    print("=== Starting Model Export and Optimization (ONNX + INT8 Quantization) ===")
    model_dir = settings.model.model_dir
    max_length = settings.model.max_length

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    onnx_path = os.path.join(model_dir, "model.onnx")
    quantized_onnx_path = os.path.join(model_dir, "model_quantized.onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Dummy inputs for ONNX trace
    dummy_text = "Where is my order?"
    inputs = tokenizer(dummy_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    print(f"Exporting PyTorch model to ONNX at: {onnx_path}")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )

    if HAS_ONNX:
        print(f"Applying INT8 Dynamic Quantization -> {quantized_onnx_path}")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_onnx_path,
            weight_type=QuantType.QUInt8
        )
        print(" ONNX INT8 Quantization finished successfully!")
    else:
        print(" Warning: onnxruntime not installed, skipped quantization stage.")

if __name__ == "__main__":
    export_to_onnx()
