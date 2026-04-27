"""
chatbot/session.py

In-memory session store for multi-turn conversations.

LangGraph itself is stateless (every .invoke() call is independent).
This module maintains the chat_history across HTTP requests so users
can ask follow-up questions naturally.

In production you'd replace this with Redis or a DB-backed store.
"""
import uuid
from typing import Optional

MAX_HISTORY_TURNS = 20   # keep last 20 user+assistant turns


_sessions: dict[str, list[dict]] = {}
# Format: { session_id: [ {role: "user"|"assistant", content: str}, ... ] }


def get_or_create(session_id: Optional[str]) -> tuple[str, list[dict]]:
    """Return (session_id, history). Creates a new session if needed."""
    if not session_id or session_id not in _sessions:
        session_id = session_id or str(uuid.uuid4())
        _sessions[session_id] = []
    return session_id, _sessions[session_id]


def append_turn(session_id: str, user_msg: str, assistant_msg: str):
    """Add a user + assistant turn to the session history."""
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    # Trim to max turns
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        _sessions[session_id] = history[-max_msgs:]


def clear(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def get_history(session_id: str) -> list[dict]:
    return list(_sessions.get(session_id, []))


def list_sessions() -> list[str]:
    return list(_sessions.keys())
