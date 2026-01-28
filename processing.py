# import json

# import os
# import re
# import io
# import json
# import pandas as pd
# from dotenv import load_dotenv
# from google import genai
# from google.genai.types import Tool, GenerateContentConfig, UrlContext
# from google.genai.errors import APIError
# import pdb
# from dotenv import load_dotenv
# load_dotenv()
# def load_feature_master_json(path: str) -> dict:
#     """
#     Loads feature master JSON and normalizes it for fast lookup
#     Structure:
#     {
#       category: {
#         normalized_feature_name: original_feature_name
#       }
#     }
#     """
#     with open(path, "r", encoding="utf-8") as f:
#         raw = json.load(f)

#     normalized = {}

#     for category, features in raw.items():
#         normalized[category] = {}

#         for item in features:
#             name = item.get("name")
#             if name:
#                 normalized[category][name.strip().lower()] = name.strip()

#     return normalized

# def build_llm_prompt(unmatched_rows, feature_master):
#     payload = []

#     for row in unmatched_rows:
#         category = row["Category"]
#         payload.append({
#             "category": category,
#             "excel_feature": row["Feature"],
#             "master_features": list(feature_master.get(category, {}).values())
#         })

#     prompt = f"""
# You are an automotive feature normalization expert.

# For each item, decide if the Excel feature has the SAME meaning
# as one of the master features under the same category.

# Rules:
# - Match ONLY if meaning is clearly the same
# - Return exact master feature name
# - If no match, return null
# - Do NOT invent new features
# - Do NOT change category

# Respond in STRICT JSON array.

# Input:
# {json.dumps(payload, indent=2)}

# Output format:
# [
#   {{
#     "excel_feature": "...",
#     "category": "...",
#     "matched_feature": "Master Feature Name OR null"
#   }}
# ]
# """
#     return prompt




# def call_llm(prompt: str) -> list:
#     """
#     Calls Gemini (new SDK) once.
#     Expects STRICT JSON array output.
#     No retries. No validation logic here.
#     """

#     client = genai.Client()

#     config = GenerateContentConfig(
#         temperature=0,     # deterministic
#         top_p=0.95,
#         top_k=40,
#         response_mime_type="application/json"
#     )

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[prompt],
#         config=config
#     )

#     text = response.text.strip()

#     # Safety: Gemini may still wrap JSON
#     if text.startswith("```"):
#         text = text.split("```")[1].strip()

#     return json.loads(text)


# def map_excel_features_with_llm(
#     excel_path: str,
#     feature_master: dict,
#     output_path: str
# ):
#     df = pd.read_excel(excel_path)

#     unmatched_rows = []

#     # STEP 1: Exact Match
#     def resolve_feature(row):
#         category = str(row["Category"]).strip()
#         feature_raw = str(row["Feature"]).strip()
#         feature_norm = feature_raw.lower()

#         if category in feature_master:
#             matched = feature_master[category].get(feature_norm)
#             if matched:
#                 return matched

#         unmatched_rows.append({
#             "row_index": row.name,
#             "Category": category,
#             "Feature": feature_raw
#         })
#         return None

#     df["master_feature_name"] = df.apply(resolve_feature, axis=1)

#     # STEP 2: LLM Matching (ONLY unmatched)
#     if unmatched_rows:
#         print(len(unmatched_rows))
#         if not os.getenv("GOOGLE_API_KEY"):
#             raise EnvironmentError("GOOGLE_API_KEY not set in .env file")
#         prompt = build_llm_prompt(unmatched_rows, feature_master)
#         llm_results = call_llm(prompt)

#         # Apply LLM results
#         for res in llm_results:
#             if res.get("matched_feature"):
#                 for row in unmatched_rows:
#                     if (
#                         row["Feature"] == res["excel_feature"]
#                         and row["Category"] == res["category"]
#                     ):
#                         df.at[row["row_index"], "master_feature_name"] = res["matched_feature"]

#     df.to_excel(output_path, index=False)

# if __name__ == "__main__":
#     feature_master = load_feature_master_json("feature_master.json")

#     map_excel_features_with_llm(
#         excel_path="creta20.xlsx",
#         feature_master=feature_master,
#         output_path="car_features_mapped.xlsx"
#     )


import os
import json
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

# ---------------- LOGGER SETUP ---------------- #
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )
# logger = logging.getLogger(__name__)


# ---------------- UTIL FUNCTIONS ---------------- #

def load_feature_master_json(path: str) -> dict:
    """
    Loads feature master JSON and normalizes it for fast lookup
    """
    start = time.time()
    # logger.info("Loading feature master JSON")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = {}
    for category, features in raw.items():
        normalized[category] = {}
        for item in features:
            name = item.get("name")
            if name:
                normalized[category][name.strip().lower()] = name.strip()

    # # logger.info(
    #     "Feature master loaded | Categories: %d | Time: %.2fs",
    #     len(normalized),
    #     time.time() - start
    # )
    return normalized


def build_llm_prompt(unmatched_rows, feature_master):
    payload = []

    for row in unmatched_rows:
        payload.append({
            "category": row["Category"],
            "excel_feature": row["Feature"],
            "master_features": list(
                feature_master.get(row["Category"], {}).values()
            )
        })

    return f"""
You are an automotive feature normalization expert.

Rules:
- Match ONLY if meaning is clearly the same
- Return exact master feature name
- If no match, return null
- Do NOT invent new features
- Do NOT change category

Respond in STRICT JSON array.

Input:
{json.dumps(payload, indent=2)}

Output:
[
  {{
    "excel_feature": "...",
    "category": "...",
    "matched_feature": "Master Feature Name OR null"
  }}
]
"""


def call_llm(prompt: str) -> list:
    """
    Calls Gemini LLM
    """
    # logger.info("LLM call started")
    start = time.time()

    client = genai.Client()
    config = GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config=config
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].strip()

    # logger.info("LLM call finished | Time: %.2fs", time.time() - start)
    return json.loads(text)


# ---------------- MAIN MAPPING FUNCTION ---------------- #

# def map_excel_features_with_llm(
#     excel_path: str,
#     feature_master: dict,
#     output_path: str
# ):
#     overall_start = time.time()
#     logger.info("Reading Excel file: %s", excel_path)

#     df = pd.read_excel(excel_path)
#     total_rows = len(df)

#     logger.info("Total rows in Excel: %d", total_rows)

#     unmatched_rows = []
#     exact_match_count = 0

#     # -------- STEP 1: Exact Match -------- #
#     def resolve_feature(row):
#         nonlocal exact_match_count

#         category = str(row["Category"]).strip()
#         feature_raw = str(row["Feature"]).strip()
#         feature_norm = feature_raw.lower()

#         if category in feature_master:
#             matched = feature_master[category].get(feature_norm)
#             if matched:
#                 exact_match_count += 1
#                 return matched

#         unmatched_rows.append({
#             "row_index": row.name,
#             "Category": category,
#             "Feature": feature_raw
#         })
#         return None

#     logger.info("Starting exact match processing")
#     df["master_feature_name"] = df.apply(resolve_feature, axis=1)

#     logger.info(
#         "Exact match completed | Matched: %d | Unmatched: %d",
#         exact_match_count,
#         len(unmatched_rows)
#     )

#     # -------- STEP 2: LLM Matching -------- #
#     if unmatched_rows:
#         if not os.getenv("GOOGLE_API_KEY"):
#             raise EnvironmentError("GOOGLE_API_KEY not set")

#         logger.info("Sending %d unmatched rows to LLM", len(unmatched_rows))
#         prompt = build_llm_prompt(unmatched_rows, feature_master)
#         llm_results = call_llm(prompt)

#         llm_match_count = 0
#         for res in llm_results:
#             if res.get("matched_feature"):
#                 llm_match_count += 1
#                 for row in unmatched_rows:
#                     if (
#                         row["Feature"] == res["excel_feature"]
#                         and row["Category"] == res["category"]
#                     ):
#                         df.at[row["row_index"], "master_feature_name"] = res["matched_feature"]

#         logger.info("LLM matched rows: %d", llm_match_count)

#     # -------- SAVE OUTPUT -------- #
#     df.to_excel(output_path, index=False)
#     logger.info("Output written to: %s", output_path)

#     logger.info("TOTAL EXECUTION TIME: %.2fs", time.time() - overall_start)


def map_excel_features_with_llm(
    excel_path: str,
    feature_master: dict,
    output_path: str
):
    overall_start = time.time()
    # logger.info("Reading Excel file: %s", excel_path)

    df = pd.read_excel(excel_path)
    total_rows = len(df)
    # logger.info("Total rows: %d", total_rows)

    unmatched_rows = []
    exact_match_count = 0

    # Preserve original feature
    df["original_name"] = df["Feature"]

    # ---------- STEP 1: Exact Match ----------
    def resolve_feature(row):
        nonlocal exact_match_count

        category = str(row["Category"]).strip()
        feature_raw = str(row["Feature"]).strip()
        feature_norm = feature_raw.lower()

        if category in feature_master:
            matched = feature_master[category].get(feature_norm)
            if matched:
                exact_match_count += 1
                return matched

        unmatched_rows.append({
            "row_index": row.name,
            "Category": category,
            "Feature": feature_raw
        })
        return None

    # logger.info("Starting exact match")
    df["master_feature_name"] = df.apply(resolve_feature, axis=1)

    # logger.info(
        # "Exact matched: %d | Unmatched: %d",
        # exact_match_count,
        # len(unmatched_rows)
    # )

    # ---------- STEP 2: LLM ----------
    if unmatched_rows:
        # logger.info("Sending %d rows to LLM", len(unmatched_rows))
        prompt = build_llm_prompt(unmatched_rows, feature_master)
        llm_results = call_llm(prompt)

        llm_match_count = 0
        for res in llm_results:
            if res.get("matched_feature"):
                llm_match_count += 1
                for row in unmatched_rows:
                    if (
                        row["Feature"] == res["excel_feature"]
                        and row["Category"] == res["category"]
                    ):
                        df.at[row["row_index"], "master_feature_name"] = res["matched_feature"]

        # logger.info("LLM matched: %d", llm_match_count)

    # ---------- STEP 3: Mapping Status ----------
    def status(row):
        return "MAPPED" if pd.notna(row["master_feature_name"]) else "REVIEW"

    df["mapping_status"] = df.apply(status, axis=1)

    # ---------- SAVE ----------
    df.to_excel(output_path, index=False)

    # logger.info("Final Excel written: %s", output_path)
    # logger.info("Total execution time: %.2fs", time.time() - overall_start)

# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    feature_master = load_feature_master_json("feature_master.json")

    map_excel_features_with_llm(
        excel_path="creta20.xlsx",
        feature_master=feature_master,
        output_path="creta_data_final.xlsx"
    )

