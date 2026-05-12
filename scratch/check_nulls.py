import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_nulls():
    conn = psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname")
    )
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM plan_features WHERE display_order IS NULL;")
    null_count = cur.fetchone()[0]
    print(f"NULL display_order count: {null_count}")
    
    if null_count > 0:
        print("Fixing NULLs...")
        cur.execute("UPDATE plan_features SET display_order = 0 WHERE display_order IS NULL;")
        conn.commit()
        print("Fixed.")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_nulls()
