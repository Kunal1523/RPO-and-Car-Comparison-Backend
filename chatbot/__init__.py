"""
chatbot package

LangGraph-powered Text-to-SQL chatbot for the Car Comparison platform.

Graph flow:
    classify_intent → generate_sql → validate_sql → execute_sql
                   ↘ general_answer               ↗ retry (max 2×)
                                                  ↘ format_response → END
"""
