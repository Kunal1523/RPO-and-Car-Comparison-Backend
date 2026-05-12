import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def migrate():
    # Attempt to use the same connection logic as DBManager
    conn_params = {
        "user": os.getenv("user"),
        "password": os.getenv("password"),
        "host": os.getenv("host"),
        "port": os.getenv("port"),
        "dbname": os.getenv("dbname")
    }
    
    print(f"Connecting to {conn_params['host']}/{conn_params['dbname']}...")
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        with conn.cursor() as cur:
            # PostgreSQL 9.6+ supports IF NOT EXISTS for ADD COLUMN
            print("Adding price_delta column if not exists...")
            cur.execute("ALTER TABLE plan_features ADD COLUMN IF NOT EXISTS price_delta NUMERIC DEFAULT 0;")
            
            # Ensure existing rows have 0 if they were NULL (though DEFAULT 0 handles new ones)
            print("Ensuring 0 values for price_delta...")
            cur.execute("UPDATE plan_features SET price_delta = 0 WHERE price_delta IS NULL;")
            
            print("Success!")
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
