import psycopg2
from psycopg2 import errors
import os
from dotenv import load_dotenv
import pdb
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import contextlib
import threading

load_dotenv()

# Shared pool for UserDBHandler
_pool1 = None
_pool_lock1 = threading.Lock()

def get_db_pool1():
    global _pool1
    if _pool1 is None:
        with _pool_lock1:
            if _pool1 is None:
                db_url = os.getenv("DATABASE_URL")
                if not db_url:
                    print("[DBmanager1] WARNING: DATABASE_URL is not set — database pool skipped.")
                    return None
                try:
                    _pool1 = pool.ThreadedConnectionPool(1, 2, db_url)
                except Exception as e:
                    print(f"[DBmanager1] WARNING: Could not connect to database — {e}")
                    return None
    return _pool1

class DbManager:
    def __init__(self):
        pass

    def get_conn(self):
        """
        Returns a context manager that checks out ONE connection from the pool
        and returns it when the `with` block exits.
        """
        class _ManagedConn:
            def __init__(self, db_pool):
                self._pool = db_pool
                self._conn = None

            def __enter__(self):
                self._conn = self._pool.getconn()
                self._conn.autocommit = True
                return self._conn

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self._conn:
                    try:
                        if not self._conn.autocommit and self._conn.status != 0:
                            self._conn.rollback()
                    except Exception:
                        pass
                    self._pool.putconn(self._conn)
                    self._conn = None
                return False  # do not suppress exceptions

        db_pool = get_db_pool1()
        if db_pool is None:
            raise RuntimeError("Database is unavailable (DATABASE_URL not set or DB unreachable).")
        return _ManagedConn(db_pool)

    @staticmethod
    def release_conn():
        """Kept as a no-op static method for backwards compatibility."""
        pass

class UserDBHandler(DbManager):

    async def save_microsoft_tokens_to_db(
        self,
        owner_email: str,
        username: str,
        access_token: str,
        refresh_token: str,
        token_expires_at
    ):
        """
        Insert or update Microsoft OAuth tokens for a user
        """

        query = """
        INSERT INTO users (
            owner_email,
            username,
            access_token,
            refresh_token,
            token_expires_at,
            source
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (owner_email, source)
        DO UPDATE SET
            username = EXCLUDED.username,
            access_token = EXCLUDED.access_token,
            refresh_token = EXCLUDED.refresh_token,
            token_expires_at = EXCLUDED.token_expires_at,
            created_at = NOW()
        RETURNING id;
        """

        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        query,
                        (
                            owner_email,
                            username,
                            access_token,
                            refresh_token,
                            token_expires_at,
                            "microsoft"
                        )
                    )
                    result = cur.fetchone()
                    return result

        except Exception as e:
            print("Error saving Microsoft tokens:", str(e))
            raise


    def get_microsoft_tokens(self, owner_email: str):
        query = """
        SELECT access_token, refresh_token, token_expires_at
        FROM users
        WHERE owner_email = %s
        AND source = 'microsoft'
        AND token_expires_at > NOW()
        """

        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (owner_email,))
                return cur.fetchone()
    
    def get_owner_email(self, x_user_email:str):
        query = """
              SELECT owner_email
              FROM users
              WHERE owner_email = %s
              LIMIT 1
              """
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (x_user_email,))
                return cur.fetchone()        
    
    def update_microsoft_tokens(
    self,
    owner_email: str,
    access_token: str,
    refresh_token: str,
    expires_at,
):
        query = """
        UPDATE users
        SET access_token = %s,
            refresh_token = %s,
            token_expires_at = %s
        WHERE owner_email = %s
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (access_token, refresh_token, expires_at, owner_email))
                conn.commit()

    def insert_feedback(self, data: dict):
        """
        Insert user feedback into the user_feedback table
        """
        query = """
        INSERT INTO user_feedback (
            email,
            feedback_text,
            page_url,
            project_type,
            timestamp
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            data.get('email'),
                            data.get('feedback_text'),
                            data.get('page_url'),
                            data.get('project_type'),
                            data.get('timestamp')
                        )
                    )
                    conn.commit()
                    return True
        except Exception as e:
            print("Error inserting feedback:", str(e))
            raise
