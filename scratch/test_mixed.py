import httpx
import json

def test_mixed_compare():
    url = "http://localhost:8000/api/compare/mixed"
    # Need to find a valid variant class and plan id
    # I'll just try to fetch some first
    try:
        plans_resp = httpx.get("http://localhost:8000/api/model-plans")
        plans = plans_resp.json().get("data", [])
        if not plans:
            print("No plans found to test.")
            return
        
        plan_id = plans[0]["plan_id"]
        variant_class = plans[0]["base_variant_class"]
        
        payload = {
            "variant_classes": [variant_class],
            "plan_ids": [plan_id],
            "version": 1
        }
        
        print(f"Testing with payload: {payload}")
        resp = httpx.post(url, json=payload, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
        else:
            print("Success!")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_mixed_compare()
