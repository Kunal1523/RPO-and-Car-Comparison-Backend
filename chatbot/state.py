"""
chatbot/state.py

Defines the LangGraph state (TypedDict) shared across all graph nodes.

In LangGraph, the State is the single source of truth that every node
reads from and writes back to. Think of it as the "context object"
that flows through the entire graph.
"""
from typing import TypedDict, Optional, Annotated
import operator


class ChatState(TypedDict):
    """
    The shared state object that flows through every node in the LangGraph.

    LangGraph copies this forward at each step, so each node receives the
    full accumulated context and returns only the keys it wants to update.

    Fields
    ------
    session_id       : unique per browser tab / user session
    user_question    : the raw question the user just asked
    chat_history     : list of {role: "user"|"assistant", content: str}
                       kept for multi-turn context in final response generation
    intent           : "db_query" – needs SQL  |  "general" – direct answer
    generated_sql    : the SQL the LLM produced (may be refined on retry)
    sql_error        : error message from the last failed DB execution
    query_results    : list[dict] rows returned by the DB
    retry_count      : how many SQL generation attempts have happened (max 2)
    final_answer     : the human-readable response to send back to the user
    """
    session_id:     str
    user_question:  str
    chat_history:   list[dict]      # [{role, content}, ...]
    intent:         str             # "db_query" | "general"
    generated_sql:  str
    sql_error:      str             # populated when execute_sql fails
    query_results:  list[dict]
    retry_count:    int
    final_answer:   str
