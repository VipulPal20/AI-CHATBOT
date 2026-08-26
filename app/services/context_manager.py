import time
import re
import logging
from typing import Dict, List, Optional, Any
from app.utils.config import settings

logger = logging.getLogger(__name__)

class SessionState:
    """Class holding session history, active context slots, and timestamps."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []
        self.active_intent: Optional[str] = None
        self.pending_slot: Optional[str] = None  # e.g., "order_id", "email"
        self.last_accessed = time.time()

    def add_message(self, sender: str, message: str, intent: str, confidence: float):
        self.history.append({
            "sender": sender,
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "timestamp": time.time()
        })
        # Maintain max context window size
        if len(self.history) > settings.context.window_size * 2:
            self.history = self.history[-settings.context.window_size * 2:]
        self.last_accessed = time.time()

class ContextManager:
    """In-memory session & conversation context manager supporting multi-turn state tracking."""
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self.ttl_seconds = settings.context.session_ttl_minutes * 60

    def get_or_create_session(self, session_id: str) -> SessionState:
        self._cleanup_expired_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        return self.sessions[session_id]

    def process_contextual_intent(self, session_id: str, raw_message: str, predicted_intent: str, confidence: float) -> Tuple[str, float]:
        """Examine session history to resolve multi-turn intent follow-ups."""
        session = self.get_or_create_session(session_id)
        cleaned = raw_message.strip()

        # Check multi-turn follow-up triggers
        # Scenario 1: Numeric input (e.g. "12345" or "order #98765") when pending_slot or last intent was order_status
        if session.pending_slot == "order_id" or (session.active_intent == "order_status" and re.match(r"^\d{4,10}$", cleaned)):
            session.pending_slot = None
            session.active_intent = "order_status"
            return "order_status", 0.99

        # Scenario 2: Email input when last intent was password_reset or account_info
        if re.match(r"[^@]+@[^@]+\.[^@]+", cleaned) and session.active_intent in ["password_reset", "account_info"]:
            session.pending_slot = None
            return session.active_intent, 0.99

        # Default fallback to model's predicted intent if confidence is reasonable
        if confidence >= settings.model.confidence_threshold:
            session.active_intent = predicted_intent
            if predicted_intent == "order_status":
                session.pending_slot = "order_id"
            else:
                session.pending_slot = None

        return predicted_intent, confidence

    def update_session(self, session_id: str, user_msg: str, bot_msg: str, intent: str, confidence: float):
        session = self.get_or_create_session(session_id)
        session.add_message("user", user_msg, intent, confidence)
        session.add_message("bot", bot_msg, intent, confidence)

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_or_create_session(session_id)
        return session.history

    def _cleanup_expired_sessions(self):
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s.last_accessed > self.ttl_seconds]
        for sid in expired:
            del self.sessions[sid]

context_manager = ContextManager()
