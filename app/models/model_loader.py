import os
import json
import logging
from typing import Dict, Tuple, Optional
from app.utils.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton lazy-loader for tokenizer, label maps, PyTorch model, and ONNX runtime session."""
    _instance: Optional["ModelLoader"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, use_onnx: bool = False):
        if self.initialized and self.use_onnx == use_onnx:
            return

        self.model_dir = settings.model.model_dir
        self.use_onnx = use_onnx
        self.label_map: Dict[int, str] = {}
        self.responses_map: Dict[str, list] = {}
        self.tokenizer = None
        self.pytorch_model = None
        self.onnx_session = None

        self._load_label_mappings()
        self._load_tokenizer()

        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

        self.initialized = True
        logger.info(f"ModelLoader initialized. Mode: {'ONNX' if use_onnx else 'PyTorch'}")

    def _load_label_mappings(self):
        label_file = os.path.join(self.model_dir, "label_encoder.json")
        resp_file = os.path.join(self.model_dir, "responses.json")

        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.label_map = {int(k): v for k, v in data.items()}

        if os.path.exists(resp_file):
            with open(resp_file, "r", encoding="utf-8") as f:
                self.responses_map = json.load(f)

    def _load_tokenizer(self):
        from app.models.tokenizer import TokenizerWrapper
        tokenizer_source = self.model_dir if os.path.exists(os.path.join(self.model_dir, "vocab.txt")) or os.path.exists(os.path.join(self.model_dir, "tokenizer.json")) else settings.model.name
        self.tokenizer = TokenizerWrapper(tokenizer_source, max_length=settings.model.max_length)

    def _load_pytorch_model(self):
        from app.models.intent_model import IntentClassifierModel
        if os.path.exists(os.path.join(self.model_dir, "config.json")):
            self.pytorch_model = IntentClassifierModel(self.model_dir)
        else:
            logger.warning(f"Trained model not found at '{self.model_dir}'. Run training first.")

    def _load_onnx_model(self):
        try:
            import onnxruntime as ort
            onnx_path = settings.model.onnx_model_path
            if not os.path.exists(onnx_path):
                onnx_path = os.path.join(self.model_dir, "model.onnx")

            if os.path.exists(onnx_path):
                self.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                logger.info(f"ONNX session loaded from {onnx_path}")
            else:
                logger.warning(f"ONNX model file not found at {onnx_path}. Falling back to PyTorch.")
                self.use_onnx = False
                self._load_pytorch_model()
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}. Falling back to PyTorch.")
            self.use_onnx = False
            self._load_pytorch_model()

model_loader = ModelLoader()
