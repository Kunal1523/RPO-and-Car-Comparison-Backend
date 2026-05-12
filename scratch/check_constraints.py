
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def check_schema():
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
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                rc.delete_rule
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.referential_constraints AS rc
                  ON tc.constraint_name = rc.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON rc.unique_constraint_name = ccu.constraint_name
                  AND rc.unique_constraint_schema = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
              AND ccu.table_name = 'model_plans';
        """
        
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            print("Foreign keys referencing model_plans:")
            for row in results:
                print(row)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
