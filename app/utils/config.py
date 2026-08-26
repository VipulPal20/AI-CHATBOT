import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3.0e-5
    confidence_threshold: float = 0.65
    model_dir: str = "models/intent_classifier"
    onnx_model_path: str = "models/intent_classifier/model_quantized.onnx"

class ContextConfig(BaseModel):
    window_size: int = 5
    session_ttl_minutes: int = 60

class DatabaseConfig(BaseModel):
    url: str = "sqlite:///chatbot.db"
    pool_size: int = 10
    max_overflow: int = 20

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"

class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    secret_key: str = "default-secret-key"

def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from config.yaml and override with environment variables."""
    cfg_dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

    config = AppConfig(**cfg_dict)

    # Environment variable overrides
    if os.getenv("DATABASE_URL"):
        config.database.url = os.getenv("DATABASE_URL")
    if os.getenv("MODEL_PATH"):
        config.model.model_dir = os.getenv("MODEL_PATH")
    if os.getenv("CONFIDENCE_THRESHOLD"):
        config.model.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
    if os.getenv("MAX_MESSAGE_LENGTH"):
        config.model.max_length = int(os.getenv("MAX_MESSAGE_LENGTH"))
    if os.getenv("CONTEXT_WINDOW"):
        config.context.window_size = int(os.getenv("CONTEXT_WINDOW"))
    if os.getenv("LOG_LEVEL"):
        config.logging.level = os.getenv("LOG_LEVEL")
    if os.getenv("SECRET_KEY"):
        config.secret_key = os.getenv("SECRET_KEY")

    return config

settings = load_config()
