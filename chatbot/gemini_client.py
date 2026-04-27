"""
chatbot/gemini_client.py

Wraps Google Gemini (google-genai SDK) for multi-turn chat.
Maintains an in-memory conversation history per session.
"""
import os
import uuid
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from chatbot.db_context import build_db_context_for_prompt, search_variants_by_name, get_variant_features

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ─────────────────────────────────────────────
# System Prompt Template
# ─────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """
You are an expert Car Comparison Assistant for an automotive research platform.
You have access to a live database containing brands, cars, variants, pricing, and feature data.

Here is the current database snapshot:
{db_context}

=== YOUR CAPABILITIES ===
- Answer questions about car variants, pricing, and features from the database above.
- Compare variants side-by-side (price, features, specs).
- Suggest best variants based on budget, fuel type, or transmission preferences.
- Explain feature differences across variants of the same or different cars.
- Give price range summaries for any car model.
- Answer general automotive knowledge questions.

=== FORMATTING RULES ===
- Always respond in clear, structured markdown.
- Use tables for comparisons (brand | variant | price | fuel).
- Use bullet points for feature lists.
- Format prices as ₹X,XX,XXX (Indian numbering).
- If you don't have the data to answer precisely, say so honestly and suggest what the user can try.
- Never fabricate prices or features that aren't in the database snapshot above.
- Be concise but thorough.

=== LANGUAGE ===
Respond in the same language the user writes in (English by default).
""".strip()


# ─────────────────────────────────────────────
# In-memory session store
# ─────────────────────────────────────────────
# Maps session_id -> list of {role, content} dicts
_sessions: dict[str, list[dict]] = {}

MAX_HISTORY_TURNS = 20   # keep last 20 turns (40 messages) to stay within context


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, list[dict]]:
    if not session_id or session_id not in _sessions:
        session_id = session_id or str(uuid.uuid4())
        _sessions[session_id] = []
    return session_id, _sessions[session_id]


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last MAX_HISTORY_TURNS * 2 messages."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        return history[-max_msgs:]
    return history


# ─────────────────────────────────────────────
# Main chat function
# ─────────────────────────────────────────────

def chat(user_message: str, session_id: Optional[str] = None) -> dict:
    """
    Send a message to Gemini with full DB context and chat history.

    Returns:
        {
            "session_id": str,
            "reply": str,
            "history_length": int
        }
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # --- Build system prompt with live DB context ---
    db_context = build_db_context_for_prompt()
    system_instruction = SYSTEM_PROMPT_TEMPLATE.format(db_context=db_context)

    # --- Session management ---
    session_id, history = _get_or_create_session(session_id)

    # Append user turn
    history.append({"role": "user", "parts": [{"text": user_message}]})
    history = _trim_history(history)

    # Build contents list for Gemini
    contents = [
        types.Content(
            role=turn["role"],
            parts=[types.Part(text=p["text"]) for p in turn["parts"]]
        )
        for turn in history
    ]

    # --- Call Gemini ---
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=2048,
        ), 
        contents=contents,
    )

    reply_text = response.text or ""

    # Append assistant turn
    history.append({"role": "model", "parts": [{"text": reply_text}]})
    _sessions[session_id] = history

    return {
        "session_id": session_id,
        "reply": reply_text,
        "history_length": len(history) // 2,  # number of full turns
    }


def clear_session(session_id: str) -> bool:
    """Clear conversation history for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def get_session_history(session_id: str) -> list[dict]:
    """Return the raw history for a session (for debugging)."""
    return _sessions.get(session_id, [])
