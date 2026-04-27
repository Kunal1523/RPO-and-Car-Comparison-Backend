"""
chatbot/session_logger.py

Per-session file logging for the car chatbot.

Every session gets its own log file:
    logs/chatbot/<date>/<session_id>.log

Each log file captures:
  - Session start / metadata
  - Every user question
  - Intent classification result
  - Generated SQL (and retry attempts)
  - SQL validation outcome
  - SQL execution result (row count / error)
  - Final answer
  - Timing for each node
  - Session end summary

Usage
-----
    from chatbot.session_logger import get_session_logger

    log = get_session_logger(session_id)
    log.user_question("What is the price of Creta?")
    log.intent("db_query")
    log.sql_generated(sql, attempt=1)
    ...
"""

import os
import logging
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ── Log root: <project_root>/logs/chatbot/<YYYY-MM-DD>/ ──────────────────────
_LOG_ROOT = Path(__file__).resolve().parent.parent / "logs" / "chatbot"


def _get_log_path(session_id: str) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = _LOG_ROOT / date_str
    # folder.mkdir(parents=True, exist_ok=True) # Disabled to prevent folder creation
    return folder / f"{session_id}.log"


def _make_file_handler(session_id: str) -> logging.FileHandler:
    log_path = _get_log_path(session_id)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    return handler


# ── Cache: one logger per session_id ─────────────────────────────────────────
_loggers: dict[str, "SessionLogger"] = {}


def get_session_logger(session_id: str) -> "SessionLogger":
    """Return (or create) the SessionLogger for this session."""
    if session_id not in _loggers:
        _loggers[session_id] = SessionLogger(session_id)
    return _loggers[session_id]


def drop_session_logger(session_id: str):
    """Call when session is cleared to release the file handler."""
    sl = _loggers.pop(session_id, None)
    if sl:
        sl.close()


# ── SessionLogger ─────────────────────────────────────────────────────────────

_DIVIDER = "─" * 72


class SessionLogger:
    """
    Structured, human-readable logger that writes to a per-session .log file.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._start = datetime.now(timezone.utc)
        self._turn = 0

        # Create a dedicated Python logger (avoids polluting the root logger)
        name = f"chatbot.session.{session_id}"
        self._log = logging.getLogger(name)
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False          # don't bubble up to uvicorn logs

        # self._handler = _make_file_handler(session_id)
        # self._log.addHandler(self._handler)
        
        # Using NullHandler to disable logging as requested by user
        self._handler = logging.NullHandler()
        self._log.addHandler(self._handler)

        # Write header
        self._write_header()

    # ── Header / footer ───────────────────────────────────────────────────────

    def _write_header(self):
        self._log.info(_DIVIDER)
        self._log.info("SESSION START")
        self._log.info("  session_id : %s", self.session_id)
        self._log.info("  started_at : %s UTC", self._start.strftime("%Y-%m-%d %H:%M:%S"))
        self._log.info("  log_file   : %s", _get_log_path(self.session_id))
        self._log.info(_DIVIDER)

    def session_end(self, total_turns: int):
        elapsed = (datetime.now(timezone.utc) - self._start).total_seconds()
        self._log.info(_DIVIDER)
        self._log.info("SESSION END")
        self._log.info("  total_turns : %d", total_turns)
        self._log.info("  duration    : %.1f s", elapsed)
        self._log.info(_DIVIDER)

    # ── Per-turn events ───────────────────────────────────────────────────────

    def new_turn(self, question: str):
        self._turn += 1
        self._log.info("")
        self._log.info("━━━  TURN %d  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", self._turn)
        self._log.info("[USER]  %s", question)

    def intent(self, intent: str):
        self._log.info("[INTENT]  → %s", intent.upper())

    def sql_generated(self, sql: str, attempt: int):
        self._log.info("[SQL GEN]  attempt=%d", attempt)
        for line in sql.splitlines():
            self._log.info("    %s", line)

    def sql_validated(self, passed: bool, reason: str = ""):
        if passed:
            self._log.info("[VALIDATE]  PASSED")
        else:
            self._log.warning("[VALIDATE]  REJECTED — %s", reason)

    def sql_executed(self, row_count: int = 0, error: str = ""):
        if error:
            self._log.error("[EXECUTE]  ERROR — %s", error)
        else:
            self._log.info("[EXECUTE]  OK — %d row(s) returned", row_count)

    def retry(self, attempt: int, error: str):
        self._log.warning("[RETRY]  attempt=%d — %s", attempt, error[:200])

    def answer(self, text: str):
        preview = textwrap.shorten(text, width=300, placeholder=" …")
        self._log.info("[ANSWER]  %s", preview)

    def error(self, msg: str, exc: Exception | None = None):
        if exc:
            self._log.exception("[ERROR]  %s", msg)
        else:
            self._log.error("[ERROR]  %s", msg)

    def info(self, msg: str):
        self._log.info("[INFO]  %s", msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self._log.info(_DIVIDER)
        self._log.info("SESSION CLEARED — logger closed")
        self._log.info(_DIVIDER)
        self._handler.close()
        self._log.removeHandler(self._handler)
