from DBManager import ChatHistoryDbManager

chat_db = ChatHistoryDbManager()

def get_or_create_session(owner_email: str, session_id = None):
    # Convert string session_id to int if it's a numeric string
    s_id = None
    if session_id is not None:
        if isinstance(session_id, int):
            s_id = session_id
        elif isinstance(session_id, str) and session_id.isdigit():
            s_id = int(session_id)
    
    return chat_db.get_or_create_session(owner_email, s_id)

def append_message(session_id: int, role: str, content: str):
    chat_db.append_message(session_id, role, content)

def get_session_history(session_id: int):
    return chat_db.get_session_history(session_id)

def list_user_sessions(owner_email: str):
    return chat_db.list_user_sessions(owner_email)

def rename_session(session_id: int, owner_email: str, new_title: str):
    chat_db.rename_session(session_id, owner_email, new_title)

def delete_session(session_id: int, owner_email: str):
    chat_db.delete_session(session_id, owner_email)

def toggle_star_session(session_id: int, owner_email: str):
    return chat_db.toggle_star_session(session_id, owner_email)
