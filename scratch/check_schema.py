import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_schema():
    conn = psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname")
    )
    cur = conn.cursor()
    
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='plan_features';")
    cols = [r[0] for r in cur.fetchall()]
    print(f"plan_features columns: {cols}")
    
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='feature_order';")
    cols_order = [r[0] for r in cur.fetchall()]
    print(f"feature_order columns: {cols_order}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_schema()
