import httpx
import json

def test_all_plans():
    url = "http://localhost:8000/api/compare/mixed"
    try:
        plans_resp = httpx.get("http://localhost:8000/api/model-plans")
        plans = plans_resp.json().get("data", [])
        if not plans:
            print("No plans found to test.")
            return
        
        for plan in plans:
            plan_id = plan["plan_id"]
            variant_class = plan["base_variant_class"]
            
            payload = {
                "variant_classes": [variant_class],
                "plan_ids": [plan_id],
                "version": 1
            }
            
            print(f"Testing plan {plan_id}...")
            resp = httpx.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"Failed for plan {plan_id}: Status {resp.status_code}")
                print(f"Error details: {resp.text}")
            else:
                print(f"Success for plan {plan_id}")
                
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_all_plans()
