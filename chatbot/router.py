"""
chatbot/router.py

FastAPI router — the HTTP layer on top of the LangGraph chatbot.

Mount in main.py:
    from chatbot.router import router as chatbot_router
    app.include_router(chatbot_router)

Endpoints:
    POST   /chatbot/chat                  → run the graph
    GET    /chatbot/session/{id}/history  → get conversation history
    DELETE /chatbot/session/{id}          → clear conversation
    GET    /chatbot/health                → liveness check
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from chatbot.graph import car_chatbot_graph
from chatbot import session as session_store
from chatbot.session_logger import get_session_logger, drop_session_logger

logger = logging.getLogger("chatbot.router")

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str            = Field(..., description="User's question")
    session_id: Optional[str] = Field(None, description="Pass back to continue a conversation")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Compare Hyundai Creta variants by price",
                "session_id": None,
            }
        }


class ChatResponse(BaseModel):
    session_id:     str
    reply:          str
    intent:         str   # "db_query" or "general" — useful for the UI
    generated_sql:  Optional[str] = None   # exposed for debugging / transparency
    db_results:     Optional[list[dict]] = None
    retry_count:    int   = 0
    history_length: int   = 0


class ClearSessionResponse(BaseModel):
    cleared:    bool
    session_id: str


class HistoryMessage(BaseModel):
    role:    str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages:   list[HistoryMessage]


# ──────────────────────────────────────────────────────────────────────────────
# POST /chatbot/chat
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the AI car comparison chatbot",
)
def send_message(payload: ChatRequest):
    """
    **AI Car Comparison Chatbot powered by LangGraph + Gemini**

    The graph flow:
    1. **classify_intent** — decides if DB lookup is needed
    2. **generate_sql** — LLM writes a SELECT query from the schema
    3. **validate_sql** — rejects non-SELECT statements (security gate)
    4. **execute_sql** — runs query on Supabase DB
    5. **retry loop** — if query fails, LLM fixes and retries (max 2×)
    6. **format_response** — LLM formats DB rows into markdown
    7. **general_answer** — for non-DB questions (no SQL involved)

    Pass `session_id` from the previous response to continue the conversation.

    **Example questions:**
    - *"What are all Hyundai Creta variants and their prices?"*
    - *"Compare Maruti Grand Vitara vs Creta petrol variants"*
    - *"Which cars have an automatic diesel variant under ₹15 lakh?"*
    - *"What features does the top-end SX(O) variant have?"*
    - *"What is ABS?"* (general — no DB needed)
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Resolve session ──────────────────────────────────────────────────────
    sid, history = session_store.get_or_create(payload.session_id)
    slog = get_session_logger(sid)          # per-session file logger
    slog.new_turn(payload.message.strip())

    # ── Build initial graph state ────────────────────────────────────────────
    initial_state = {
        "session_id":    sid,
        "user_question": payload.message.strip(),
        "chat_history":  history,     # history from previous turns
        "intent":        "",
        "generated_sql": "",
        "sql_error":     "",
        "query_results": [],
        "retry_count":   0,
        "final_answer":  "",
    }

    # ── Run the LangGraph ────────────────────────────────────────────────────
    try:
        final_state = car_chatbot_graph.invoke(initial_state)
    except Exception as e:
        slog.error("Graph execution failed", exc=e)
        logger.exception("[router] Graph execution failed")
        raise HTTPException(status_code=500, detail=f"Graph error: {str(e)}")

    answer      = final_state.get("final_answer", "Sorry, I could not generate an answer.")
    intent      = final_state.get("intent", "unknown")
    sql         = final_state.get("generated_sql", "")
    retry_count = final_state.get("retry_count", 0)

    # ── Log turn summary ─────────────────────────────────────────────────────
    slog.intent(intent)
    if sql and sql != "CANNOT_ANSWER":
        slog.sql_generated(sql, attempt=retry_count)
    slog.answer(answer)

    # ── Persist this turn in history ─────────────────────────────────────────
    session_store.append_turn(sid, payload.message.strip(), answer)

    logger.info(
        "[router] session=%s intent=%s retries=%d answer_len=%d",
        sid, intent, retry_count, len(answer),
    )

    return ChatResponse(
        session_id    = sid,
        reply         = answer,
        intent        = intent,
        generated_sql = sql if sql and sql != "CANNOT_ANSWER" else None,
        db_results    = final_state.get("query_results") if intent == "db_query" else None,
        retry_count   = retry_count,
        history_length= len(session_store.get_history(sid)) // 2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /chatbot/session/{id}/history
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
)
def get_history(session_id: str):
    """Returns the full message history for a session (useful for UI replay)."""
    msgs = session_store.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(**m) for m in msgs],
    )


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /chatbot/session/{id}
# ──────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    response_model=ClearSessionResponse,
    summary="Clear a conversation session",
)
def delete_session(session_id: str):
    """Clears history for the given session. Use to start fresh."""
    slog = get_session_logger(session_id)
    slog.session_end(total_turns=len(session_store.get_history(session_id)) // 2)
    drop_session_logger(session_id)          # close & flush the log file
    cleared = session_store.clear(session_id)
    return ClearSessionResponse(cleared=cleared, session_id=session_id)


# ──────────────────────────────────────────────────────────────────────────────
# GET /chatbot/health
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/health", summary="Chatbot health check")
def chatbot_health():
    """Confirms the chatbot router and graph are loaded."""
    return {
        "status":  "ok",
        "module":  "chatbot (LangGraph + Gemini Text-to-SQL)",
        "engine":  "gemini-3-flash-preview",
        "max_sql_retries": 2,
    }
