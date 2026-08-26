import os
from transformers import AutoTokenizer

class TokenizerWrapper:
    """Wrapper around Hugging Face tokenizer with padding, truncation, and error handling."""
    def __init__(self, model_dir_or_name: str, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)

    def encode(self, text: str):
        """Tokenize single text string returning PyTorch tensors."""
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

    def encode_numpy(self, text: str):
        """Tokenize text string returning NumPy arrays for ONNX runtime."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np"
        )
        return {
            "input_ids": encoded["input_ids"].astype("int64"),
            "attention_mask": encoded["attention_mask"].astype("int64")
        }
