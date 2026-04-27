"""
chatbot/nodes.py

All LangGraph node functions for the car comparison chatbot.

Each node is a plain Python function:
    def my_node(state: ChatState) -> dict:
        ...
        return {"key": new_value}  # only return what changed

LangGraph merges these partial dicts back into the shared state automatically.

Node execution order is controlled by the graph edges defined in graph.py.
"""
import os
import re
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from google import genai
from google.genai import types

from chatbot.state import ChatState
from chatbot.db_schema import SCHEMA, SCHEMA_SHORT
from chatbot.session_logger import get_session_logger
from chatbot.db_context import build_db_context_for_prompt

load_dotenv()

logger = logging.getLogger("chatbot.nodes")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_RETRIES = 2
MAX_ROWS = 5000   # increased to allow better data analysis as requested


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _llm(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    """
    Thin wrapper around the Google Gemini API.
    Uses gemini-2.0-flash for speed and cost efficiency.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
        max_output_tokens=2048,
    )
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config=config,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
    )
    return (response.text or "").strip()


def _get_db_conn():
    """Open a fresh psycopg2 connection from env vars."""
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def _build_history_text(history: list[dict], max_turns: int = 5) -> str:
    """
    Convert the last N conversation turns into plain text for LLM context.
    Keeps the prompt focused without blowing up token count.
    """
    relevant = history[-(max_turns * 2):]  # last N full turns
    lines = []
    for msg in relevant:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — classify_intent
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(state: ChatState) -> dict:
    """
    Determines whether the user's question requires a database lookup
    or can be answered as general automotive knowledge.

    Returns {"intent": "db_query"} or {"intent": "general"}

    Why this node? If we always run SQL generation for "What is ABS?",
    we waste tokens and risk confusing the model. Routing up-front keeps
    each downstream node focused on its job.
    """
    question = state["user_question"]
    history_text = _build_history_text(state.get("chat_history", []))

    prompt = f"""You are an intent classifier for a car comparison chatbot.
Classify the user's question into exactly one of these two categories:

  db_query  → The question needs real data from the database:
              prices, variants, features, comparisons, availability, specs of specific cars.

  general   → Can be answered from general automotive knowledge, no DB needed:
              "What is ABS?", "How does a CVT work?", "Tell me about SUVs in general."

Recent conversation:
{history_text or "(none)"}

User question: {question}

Respond with ONLY one word: db_query OR general"""

    intent_raw = _llm(prompt, temperature=0.0).strip().lower()
    intent = "db_query" if "db_query" in intent_raw else "general"

    logger.info("[classify_intent] question=%r → intent=%s", question, intent)
    get_session_logger(state["session_id"]).intent(intent)
    return {"intent": intent}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2a — generate_sql
# ══════════════════════════════════════════════════════════════════════════════

def generate_sql(state: ChatState) -> dict:
    """
    Asks the LLM to write a PostgreSQL SELECT statement based on:
      - The DB schema (not actual data)
      - The user's question
      - Any prior SQL error (for retry attempts)

    On first attempt: uses the full schema for maximum accuracy.
    On retries:       includes the previous SQL + error + short schema
                      so the model can fix exactly what went wrong.
    """
    question    = state["user_question"]
    retry_count = state.get("retry_count", 0)
    prior_sql   = state.get("generated_sql", "")
    sql_error   = state.get("sql_error", "")
    history_text = _build_history_text(state.get("chat_history", []))

    if retry_count == 0:
        db_context = build_db_context_for_prompt()
        # ── First attempt: full schema ──────────────────────────────────────
        system = f"""You are an expert PostgreSQL query writer for a Supabase database.

{SCHEMA}

DATABASE CONTEXT (Available values):
{db_context}

RULES YOU MUST FOLLOW:
1. Only write SELECT statements. Never INSERT, UPDATE, DELETE, DROP, or any DDL.
2. Always add WHERE is_latest = true for variants, pricing, and variant_features.
3. Use table aliases for readability (e.g. v for variants, p for pricing).
4. Do not artificially limit rows unless the question asks for it (e.g., "top 5").
5. Return ONLY the raw SQL — no markdown, no explanation, no backticks.
6. If the question cannot be answered from this schema, reply exactly: CANNOT_ANSWER
7. Ensure the SQL query is COMPLETE and ends with a semicolon. Never leave a statement unfinished.
8. DESCRIPTIVE SELECTS & ALIASES: Always include the feature name (`fm.name`) and the actual value (`vf.value`) in your SELECT clause if the question is about specific features. YOU MUST use explicit aliases for columns with the same name (e.g. `v.name AS variant_name`, `c.name AS car_name`, `fm.name AS feature_name`) to prevent them from overwriting each other in the JSON results.
9. EXACT MATCHING: Always use the exact brand names and car names provided in the DATABASE CONTEXT above (e.g., use 'Maruti' instead of 'Maruti Suzuki' if 'Maruti' is what appears in the context).
10. DISTINCT: Always use `SELECT DISTINCT` instead of just `SELECT` to avoid returning duplicate rows.
11. TRANSMISSION TYPES: The database uses short forms for transmissions. If the user asks for 'manual', match 'MT' (e.g., `p.transmission_type = 'MT'`). If they ask for 'automatic', match 'AT', 'AMT', 'CVT', 'DCT', etc. using an `IN` clause or `ILIKE`.
"""
        prompt = f"""Recent conversation:
{history_text or "(none)"}

User question: {question}

Write a single PostgreSQL SELECT query to answer this question."""

    else:
        # ── Retry: tell the model what broke ───────────────────────────────
        system = f"""You are an expert PostgreSQL query writer fixing a broken query.

{SCHEMA_SHORT}

RULES: Only SELECT statements. Always is_latest=true for variants/pricing/variant_features.
Return ONLY the raw SQL — no markdown, no backticks, no explanation.
Always use SELECT DISTINCT to avoid duplicates. Handle short forms for transmissions (e.g., 'MT' for manual, 'AT/AMT/CVT' for automatic).
"""
        prompt = f"""The previous SQL query failed with this error:
ERROR: {sql_error}

Previous (broken) SQL:
{prior_sql}

User's original question: {question}

Fix the SQL query so it runs without errors. Return ONLY the corrected SQL."""

    raw = _llm(prompt, system=system, temperature=0.0)

    # Strip markdown code fences if the model adds them despite instructions
    sql = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    logger.info("[generate_sql] attempt=%d sql=%s", retry_count + 1, sql[:120])
    get_session_logger(state["session_id"]).sql_generated(sql, attempt=retry_count + 1)
    return {
        "generated_sql": sql,
        "retry_count": retry_count + 1,
        "sql_error": "",           # reset error for this new attempt
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2b — validate_sql
# ══════════════════════════════════════════════════════════════════════════════

_FORBIDDEN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE | re.MULTILINE,
)

def validate_sql(state: ChatState) -> dict:
    """
    Safety gate — rejects anything that is not a SELECT statement.

    Why not rely on the LLM? LLMs can hallucinate. We enforce this
    deterministically so no destructive SQL can ever reach the DB.

    Returns {"sql_error": ""} if valid, or sets sql_error to a rejection message.
    Also handles the CANNOT_ANSWER sentinel from generate_sql.
    """
    sql = state.get("generated_sql", "").strip()
    slog = get_session_logger(state["session_id"])

    # Model said it can't answer
    if sql.upper() == "CANNOT_ANSWER" or not sql:
        slog.sql_validated(passed=False, reason="CANNOT_ANSWER")
        return {
            "sql_error": "CANNOT_ANSWER",
            "generated_sql": sql,
        }

    # Must start with SELECT or WITH
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        msg = (
            f"SECURITY: Only SELECT or WITH statements are allowed. "
            f"Rejected statement starting with: '{sql.split()[0].upper()}'"
        )
        logger.warning("[validate_sql] REJECTED: %s", msg)
        slog.sql_validated(passed=False, reason=msg)
        return {"sql_error": msg, "generated_sql": sql}

    # Check for embedded forbidden commands (e.g. SELECT ...; DROP TABLE)
    # Strip the leading SELECT or WITH before checking
    without_select = re.sub(r"^\s*(SELECT|WITH)\b", "", sql, count=1, flags=re.IGNORECASE)
    if _FORBIDDEN.search(without_select):
        msg = "SECURITY: SQL contains forbidden commands. Rejected."
        logger.warning("[validate_sql] REJECTED embedded command: %s", sql[:100])
        slog.sql_validated(passed=False, reason=msg)
        return {"sql_error": msg, "generated_sql": sql}

    logger.info("[validate_sql] SQL passed validation")
    slog.sql_validated(passed=True)
    return {"sql_error": ""}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — execute_sql
# ══════════════════════════════════════════════════════════════════════════════

def execute_sql(state: ChatState) -> dict:
    """
    Runs the validated SQL against the Supabase/PostgreSQL database.

    On success: returns {"query_results": [...rows...], "sql_error": ""}
    On failure: returns {"sql_error": "<pg error>", "query_results": []}
                The graph's conditional edge will decide whether to retry.
    """
    sql = state["generated_sql"]
    slog = get_session_logger(state["session_id"])
    conn = None
    try:
        conn = _get_db_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchmany(MAX_ROWS)
            results = [dict(r) for r in rows]

        logger.info("[execute_sql] OK — %d rows returned", len(results))
        slog.sql_executed(row_count=len(results))
        return {"query_results": results, "sql_error": ""}

    except Exception as e:
        err = str(e).strip()
        logger.warning("[execute_sql] ERROR: %s", err)
        slog.sql_executed(error=err)
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_RETRIES:
            slog.retry(attempt=retry_count + 1, error=err)
        return {"query_results": [], "sql_error": err}

    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4a — format_response   (after DB query)
# ══════════════════════════════════════════════════════════════════════════════

def format_response(state: ChatState) -> dict:
    """
    Takes raw DB rows and turns them into a well-formatted, human-readable answer.

    This node handles three sub-cases:
      A) Normal: has query_results → format as markdown table / bullets
      B) CANNOT_ANSWER sentinel → politely explain and suggest alternatives
      C) Permanent SQL failure (exceeded retries) → tell user what went wrong
    """
    question     = state["user_question"]
    results      = state.get("query_results", [])
    sql_error    = state.get("sql_error", "")
    generated_sql = state.get("generated_sql", "")
    history_text = _build_history_text(state.get("chat_history", []))

    # ── Case B: model said it cannot answer from schema ──────────────────────
    if sql_error == "CANNOT_ANSWER":
        answer = (
            "I couldn't find specific data in the database to answer your question. "
            "You can ask me about:\n"
            "- Prices of specific car variants (e.g. *Hyundai Creta prices*)\n"
            "- Feature comparisons between variants\n"
            "- Which cars fall within a budget\n"
            "- Diesel / Petrol / Automatic availability"
        )
        return {"final_answer": answer}

    # ── Case C: SQL hard failure (retries exhausted) ─────────────────────────
    if sql_error and not results:
        answer = (
            f"I ran into a database error while fetching your answer.\n\n"
            f"**Error:** `{sql_error}`\n\n"
            f"Please try rephrasing your question or be more specific about the car/variant name."
        )
        return {"final_answer": answer}

    # ── Case A: we have results → ask LLM to format them ────────────────────
    # Serialize results to a JSON string (compact, readable)
    results_text = json.dumps(results, indent=2, default=str)

    system = """You are a helpful car comparison assistant.
Your goal is to provide an accurate, data-driven answer based ONLY on the database results provided.

STRICT GROUNDING RULES:
1. ONLY use information from the "Database results (raw JSON)" provided below.
2. NEVER use general knowledge about cars to supplement the answer. If a price or feature is not in the JSON, do not mention it.
3. If the "Database results" are empty ([]), you MUST say: "I couldn't find any matching data in my database for your specific request."
4. After a "No data found" message, you may suggest alternative queries based on the schema categories (e.g., "You can try asking for variants in a different price range or check specific features like ABS or Airbags").
5. Format prices exactly as ₹X,XX,XXX or in Lakhs.
6. Use markdown tables for comparisons and bold for key values."""

    prompt = f"""Recent conversation:
{history_text or "(none)"}

User question: {question}

SQL that was executed:
{generated_sql}

Database results (raw JSON):
{results_text}

Format the results into a helpful, well-structured markdown response for the user."""

    answer = _llm(prompt, system=system, temperature=0.3)
    logger.info("[format_response] answer generated (%d chars)", len(answer))
    get_session_logger(state["session_id"]).answer(answer)
    return {"final_answer": answer}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4b — general_answer   (no DB needed)
# ══════════════════════════════════════════════════════════════════════════════

def general_answer(state: ChatState) -> dict:
    """
    Answers general automotive questions without touching the database.
    Examples: "What is ABS?", "How does a CVT work?", "What is a sunroof?"

    This keeps DB queries focused while still making the chatbot useful
    for educational questions.
    """
    question     = state["user_question"]
    history_text = _build_history_text(state.get("chat_history", []))

    system = """You are a knowledgeable car industry expert assistant.
Answer the user's automotive question clearly and concisely.
Use markdown formatting where appropriate (bold, bullet points).
Keep answers focused and practical."""

    prompt = f"""Recent conversation:
{history_text or "(none)"}

User question: {question}

Answer helpfully."""

    answer = _llm(prompt, system=system, temperature=0.4)
    logger.info("[general_answer] answered general question")
    get_session_logger(state["session_id"]).answer(answer)
    return {"final_answer": answer}
