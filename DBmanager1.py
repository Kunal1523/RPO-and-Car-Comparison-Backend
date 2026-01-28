import psycopg2
from psycopg2 import errors
import os
from dotenv import load_dotenv
import pdb
from psycopg2.extras import RealDictCursor
from psycopg2.extras import RealDictCursor
from psycopg2 import errors
load_dotenv()


class DbManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            os.getenv("DATABASE_URL")
        )
        self.conn.autocommit = True




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
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
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

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (owner_email,))
            return cur.fetchone()
    
    def get_owner_email(self, x_user_email:str):
        query = """
              SELECT owner_email
              FROM users
              WHERE owner_email = %s
              LIMIT 1
              """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
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
        with self.conn.cursor() as cur:
            cur.execute(query, (access_token, refresh_token, expires_at, owner_email))
            self.conn.commit()
