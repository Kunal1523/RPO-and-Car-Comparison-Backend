import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def migrate():
    conn = psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname")
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        print("Adding price_delta column...")
        try:
            cur.execute("ALTER TABLE plan_features ADD COLUMN price_delta NUMERIC DEFAULT 0;")
            print("Success!")
        except Exception as e:
            print(f"Error or already exists: {e}")
    conn.close()

if __name__ == "__main__":
    migrate()
