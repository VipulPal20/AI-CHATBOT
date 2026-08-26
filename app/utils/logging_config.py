import logging
import json
import sys
from datetime import datetime

class StructuredJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured application logging."""
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_object["request_id"] = record.request_id
        if hasattr(record, "intent"):
            log_object["intent"] = record.intent
        if hasattr(record, "confidence"):
            log_object["confidence"] = record.confidence
        if hasattr(record, "latency_ms"):
            log_object["latency_ms"] = record.latency_ms
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_object)

def setup_logging(level_name: str = "INFO"):
    """Configure logger with structured formatting."""
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(StructuredJsonFormatter())
    logger.addHandler(handler)
    return logger
