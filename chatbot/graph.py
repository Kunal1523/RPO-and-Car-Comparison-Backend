"""
chatbot/graph.py

Builds and compiles the LangGraph StateGraph for the car chatbot.

══════════════════════════════════════════════════════════════════
  GRAPH ARCHITECTURE (Text-to-SQL with retry loop)
══════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────┐
    │                     START                       │
    └────────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   classify_intent    │  "Is this a DB question?"
              └──────────┬───────────┘
                         │
           ┌─────────────┴──────────────┐
           │ "db_query"                 │ "general"
           ▼                            ▼
  ┌─────────────────┐        ┌─────────────────────┐
  │  generate_sql   │        │   general_answer     │
  │ (LLM→SQL)       │        │ (no DB needed)       │
  └────────┬────────┘        └──────────┬───────────┘
           │                            │
           ▼                            │
  ┌─────────────────┐                  │
  │  validate_sql   │  ← safety gate   │
  │ (SELECT only?)  │                  │
  └────────┬────────┘                  │
           │                           │
     ┌─────┴──────┐                    │
     │ "invalid"  │ "valid"            │
     │    or      │                    │
     │ "cannot"   ▼                    │
     │    ┌──────────────┐             │
     │    │  execute_sql │             │
     │    │  (run on DB) │             │
     │    └──────┬───────┘             │
     │           │                     │
     │    ┌──────┴──────────────┐      │
     │    │ "success"  "error"  │      │
     │    │            │        │      │
     │    │      retry_count    │      │
     │    │     < MAX_RETRIES?  │      │
     │    │     yes ──► generate_sql   │
     │    │     no  ──┐         │      │
     │    │           │         │      │
     ▼    ▼           ▼         │      │
  ┌──────────────────────────┐  │      │
  │      format_response     │◄─┘      │
  │  (LLM formats DB rows)   │         │
  └──────────┬───────────────┘         │
             │                         │
             └─────────────┬───────────┘
                           ▼
                         END


Key LangGraph concepts used here:
  • StateGraph   — the graph container
  • TypedDict    — defines the shared state schema
  • add_node()   — registers each processing function
  • add_edge()   — unconditional arrows between nodes
  • add_conditional_edges() — branching based on state values
  • START / END  — built-in graph sentinels
  • compile()    — locks the graph and returns a runnable
══════════════════════════════════════════════════════════════════
"""
import logging
from langgraph.graph import StateGraph, START, END

from chatbot.state import ChatState
from chatbot.nodes import (
    classify_intent,
    generate_sql,
    validate_sql,
    execute_sql,
    format_response,
    general_answer,
)

logger = logging.getLogger("chatbot.graph")

MAX_RETRIES = 2   # must match nodes.py


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# These return a string key that LangGraph uses to pick the next node.
# ──────────────────────────────────────────────────────────────────────────────

def route_after_intent(state: ChatState) -> str:
    """Branch after classify_intent."""
    return state["intent"]   # "db_query" | "general"


def route_after_validate(state: ChatState) -> str:
    """
    After validate_sql:
      - "valid"    → proceed to execute_sql
      - "invalid"  → skip to format_response (will show security error)
    """
    err = state.get("sql_error", "")
    if not err:
        return "valid"
    return "invalid"     # covers CANNOT_ANSWER + security rejections


def route_after_execute(state: ChatState) -> str:
    """
    After execute_sql:
      - "success"  → format_response (has real rows)
      - "retry"    → generate_sql again with the error context
      - "failed"   → format_response (retries exhausted, show error)
    """
    err  = state.get("sql_error", "")
    done = state.get("retry_count", 0) >= MAX_RETRIES

    if not err:
        return "success"
    if done:
        return "failed"
    return "retry"


# ──────────────────────────────────────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph.
    Call once at startup; the compiled graph is thread-safe and reusable.
    """
    builder = StateGraph(ChatState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("generate_sql",    generate_sql)
    builder.add_node("validate_sql",    validate_sql)
    builder.add_node("execute_sql",     execute_sql)
    builder.add_node("format_response", format_response)
    builder.add_node("general_answer",  general_answer)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "classify_intent")

    # ── Route after intent classification ─────────────────────────────────────
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "db_query": "generate_sql",
            "general":  "general_answer",
        },
    )

    # ── DB path: generate → validate → execute ────────────────────────────────
    builder.add_edge("generate_sql", "validate_sql")

    builder.add_conditional_edges(
        "validate_sql",
        route_after_validate,
        {
            "valid":   "execute_sql",
            "invalid": "format_response",   # security error or CANNOT_ANSWER
        },
    )

    builder.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "success": "format_response",
            "retry":   "generate_sql",      # loops back with error context
            "failed":  "format_response",
        },
    )

    # ── Both paths end at END ─────────────────────────────────────────────────
    builder.add_edge("format_response", END)
    builder.add_edge("general_answer",  END)

    graph = builder.compile()
    logger.info("[graph] LangGraph compiled successfully")
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton (compiled once at import time)
# ──────────────────────────────────────────────────────────────────────────────
car_chatbot_graph = build_graph()
