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
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List

from chatbot.graph import car_chatbot_graph
from chatbot.nodes import summarize_chat_title
from chatbot import session_db
from chatbot.session_logger import get_session_logger, drop_session_logger
from dependencies import get_current_user

logger = logging.getLogger("chatbot.router")

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str            = Field(..., description="User's question")
    session_id: Optional[int] = Field(None, description="Pass back to continue a conversation")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Compare Hyundai Creta variants by price",
                "session_id": None,
            }
        }


class ChatResponse(BaseModel):
    session_id:     int
    reply:          str
    intent:         str   # "db_query" or "general" — useful for the UI
    generated_sql:  Optional[str] = None   # exposed for debugging / transparency
    db_results:     Optional[list[dict]] = None
    retry_count:    int   = 0
    history_length: int   = 0


class ClearSessionResponse(BaseModel):
    cleared:    bool
    session_id: int


class HistoryMessage(BaseModel):
    role:    str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: int
    messages:   list[HistoryMessage]

class SessionInfo(BaseModel):
    id: int
    title: str
    is_starred: bool = False
    updated_at: int
    created_at: int

class SessionRenameRequest(BaseModel):
    title: str


# ──────────────────────────────────────────────────────────────────────────────
# POST /chatbot/chat
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the AI car comparison chatbot",
)
def send_message(
    payload: ChatRequest, 
    user_email: str = Depends(get_current_user)
):
    """
    **AI Car Comparison Chatbot powered by LangGraph + Gemini**
    Now supports persistent, user-based chat history.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Resolve session (DB backed) ──────────────────────────────────────────
    chat_session = session_db.get_or_create_session(user_email, payload.session_id)
    sid = chat_session["id"]
    history = session_db.get_session_history(sid)
    
    slog = get_session_logger(sid)
    slog.new_turn(payload.message.strip())

    # ── Build initial graph state ────────────────────────────────────────────
    initial_state = {
        "session_id":    sid,
        "user_question": payload.message.strip(),
        "chat_history":  history,
        "intent":        "",
        "generated_sql": "",
        "sql_error":     "",
        "query_results": [],
        "retry_count":   0,
        "final_answer":  "",
    }

    # ── Release DB connection before long AI task ─────────────────────────────
    # This prevents "Pool Exhausted" errors while waiting for Gemini
    from DBManager import DbManager as ChatDbManager
    from DBmanager1 import DbManager as UserDbManager
    ChatDbManager.release_conn()
    UserDbManager.release_conn()

    # ── Run the LangGraph ────────────────────────────────────────────────────
    try:
        final_state = car_chatbot_graph.invoke(initial_state)
    except Exception as e:
        slog.error("Graph execution failed", exc=e)
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

    # ── Persist this turn in DB history ──────────────────────────────────────
    session_db.append_message(sid, "user", payload.message.strip())
    session_db.append_message(sid, "assistant", answer)

    # Auto-rename "New Chat" sessions if this is the first turn
    if chat_session["title"] == "New Chat":
        try:
            new_title = summarize_chat_title(payload.message.strip())
            session_db.rename_session(sid, user_email, new_title)
        except Exception:
            # Fallback to simple snippet if LLM call fails
            new_title = (payload.message.strip()[:47] + '...') if len(payload.message.strip()) > 50 else payload.message.strip()
            session_db.rename_session(sid, user_email, new_title)

    return ChatResponse(
        session_id    = sid,
        reply         = answer,
        intent        = intent,
        generated_sql = sql if sql and sql != "CANNOT_ANSWER" else None,
        db_results    = final_state.get("query_results") if intent == "db_query" else None,
        retry_count   = retry_count,
        history_length= len(session_db.get_session_history(sid)) // 2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /chatbot/sessions
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/sessions",
    response_model=List[SessionInfo],
    summary="List all chat sessions for the current user",
)
def list_sessions(user_email: str = Depends(get_current_user)):
    sessions = session_db.list_user_sessions(user_email)
    return [
        SessionInfo(
            id=s["id"], 
            title=s["title"] or "Untitled Chat", 
            is_starred=s.get("is_starred", False),
            updated_at=s["updated_at"], 
            created_at=s["created_at"]
        ) 
        for s in sessions
    ]


# ──────────────────────────────────────────────────────────────────────────────
# PATCH /chatbot/session/{id}
# ──────────────────────────────────────────────────────────────────────────────

@router.patch(
    "/session/{session_id}",
    summary="Rename a chat session",
)
def rename_session(
    session_id: int, 
    payload: SessionRenameRequest, 
    user_email: str = Depends(get_current_user)
):
    session_db.rename_session(session_id, user_email, payload.title)
    return {"status": "success", "session_id": session_id, "new_title": payload.title}


@router.post(
    "/session/{session_id}/star",
    summary="Toggle star status for a session",
)
def toggle_star(
    session_id: int,
    user_email: str = Depends(get_current_user)
):
    is_starred = session_db.toggle_star_session(session_id, user_email)
    return {"status": "success", "session_id": session_id, "is_starred": is_starred}


# ──────────────────────────────────────────────────────────────────────────────
# GET /chatbot/session/{id}/history
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
)
def get_history(
    session_id: int, 
    user_email: str = Depends(get_current_user)
):
    """Returns the full message history for a session from the database."""
    msgs = session_db.get_session_history(session_id)
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
    summary="Clear/Delete a conversation session",
)
def delete_session(
    session_id: int, 
    user_email: str = Depends(get_current_user)
):
    """Deletes a session and all its messages from the database."""
    slog = get_session_logger(str(session_id))
    msgs = session_db.get_session_history(session_id)
    slog.session_end(total_turns=len(msgs) // 2)
    drop_session_logger(str(session_id))
    
    session_db.delete_session(session_id, user_email)
    return ClearSessionResponse(cleared=True, session_id=session_id)


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
