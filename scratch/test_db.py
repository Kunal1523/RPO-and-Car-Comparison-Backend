
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def test_db():
    try:
        conn = psycopg2.connect(
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port"),
            dbname=os.getenv("dbname")
        )
        print("Connection successful")
        
        query = """
            SELECT 
                b.id as brand_id,
                b.name as brand_name,
                c.id as car_id,
                c.name as car_name
            FROM brands b
            LEFT JOIN cars c ON b.id = c.brand_id
            ORDER BY b.name, c.name
            LIMIT 5;
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()
            print(f"Query successful, found {len(results)} rows")
            for row in results:
                print(row)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_db()
