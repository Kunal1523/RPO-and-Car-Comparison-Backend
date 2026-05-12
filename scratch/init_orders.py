import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def init_display_orders():
    conn = psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname")
    )
    cur = conn.cursor()
    
    # Get all distinct plan_id and categories
    cur.execute("SELECT DISTINCT plan_id, category FROM plan_features;")
    groups = cur.fetchall()
    
    for plan_id, category in groups:
        # Fetch features in this group ordered by id (or feature_name)
        cur.execute("""
            SELECT id FROM plan_features 
            WHERE plan_id = %s AND category = %s
            ORDER BY id ASC;
        """, (plan_id, category))
        features = cur.fetchall()
        
        # Update display_order sequentially
        for idx, (feat_id,) in enumerate(features):
            cur.execute("""
                UPDATE plan_features
                SET display_order = %s
                WHERE id = %s
            """, (idx * 10, feat_id))  # Using * 10 to leave gaps for easy insertion later if needed
            
    conn.commit()
    print("display_order initialized.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_display_orders()
