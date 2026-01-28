import json

import os
import re
import io
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import Tool, GenerateContentConfig, UrlContext
from google.genai.errors import APIError
import pdb
from dotenv import load_dotenv
load_dotenv()
def load_feature_master_json(path: str) -> dict:
    """
    Loads feature master JSON and normalizes it for fast lookup
    Structure:
    {
      category: {
        normalized_feature_name: original_feature_name
      }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = {}

    for category, features in raw.items():
        normalized[category] = {}

        for item in features:
            name = item.get("name")
            if name:
                normalized[category][name.strip().lower()] = name.strip()

    return normalized


def map_excel_features_with_llm(
    excel_path: str,
    feature_master: dict,
    output_path: str
):
    df = pd.read_excel(excel_path)

    unmatched_rows = []

    # STEP 1: Exact Match
    def resolve_feature(row):
        category = str(row["Category"]).strip()
        feature_raw = str(row["Feature"]).strip()
        feature_norm = feature_raw.lower()

        if category in feature_master:
            matched = feature_master[category].get(feature_norm)
            if matched:
                return matched

        unmatched_rows.append({
            "row_index": row.name,
            "Category": category,
            "Feature": feature_raw
        })
        return None

    df["master_feature_name"] = df.apply(resolve_feature, axis=1)

    # # STEP 2: LLM Matching (ONLY unmatched)
    # if unmatched_rows:
    #     print(len(unmatched_rows))
    #     if not os.getenv("GOOGLE_API_KEY"):
    #         raise EnvironmentError("GOOGLE_API_KEY not set in .env file")
    #     prompt = build_llm_prompt(unmatched_rows, feature_master)
    #     llm_results = call_llm(prompt)

    #     # Apply LLM results
    #     for res in llm_results:
    #         if res.get("matched_feature"):
    #             for row in unmatched_rows:
    #                 if (
    #                     row["Feature"] == res["excel_feature"]
    #                     and row["Category"] == res["category"]
    #                 ):
    #                     df.at[row["row_index"], "master_feature_name"] = res["matched_feature"]

    df.to_excel(output_path, index=False)

if __name__ == "__main__":
    feature_master = load_feature_master_json("feature_master.json")

    map_excel_features_with_llm(
        excel_path="creta_data_final_db2.xlsx",
        feature_master=feature_master,
        output_path="car_creta_features_mapped.xlsx"
    )
