import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def create_order_table():
    conn = psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname")
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Create feature_order table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feature_order (
            id SERIAL PRIMARY KEY,
            feature_name TEXT NOT NULL,
            category TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            model_ids UUID[] DEFAULT '{}',
            UNIQUE (feature_name, category)
        );
    """)
    
    # Check if plan_features has display_order column, if not add it for convenience
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='plan_features' AND column_name='display_order';")
    if not cur.fetchone():
        cur.execute("ALTER TABLE plan_features ADD COLUMN display_order INTEGER DEFAULT 0;")

    print("Table feature_order created and plan_features updated.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    create_order_table()
