import os
import json
import time
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# ============================================================
# ENV + LOGGER
# ============================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# RULE ENGINE
# ============================================================

IGNORE_KEYWORDS = [
    "color", "painted", "silver", "black", "chrome",
    "matte", "finish", "dual tone", "red"
]

MERGE_RULES = {
    r"auto fold|electric folding": "Electrically-Foldable ORVM",
    r"foldable key|keyless": "Remote Keyless Entry",
    r"push button start|engine start": "Engine Push Start/Stop with Smart Key",
    r"seat belt reminder": "Seat Belt Reminder (Front & Rear Seats)",
}

NEW_FEATURE_HINTS = [
    "assist", "warning", "monitor", "control", "system"
]



def auto_decide(original_name: str, mapped_master):
    """
    Decide final action for REVIEW rows
    SAFE for pandas NaN values
    """
    text = str(original_name).lower().strip()

    # 1️⃣ IGNORE cosmetic variants
    for kw in IGNORE_KEYWORDS:
        COSMETIC_ONLY_KEYWORDS = [
    "painted", "silver", "black", "chrome", "matte", "finish",
    "dual tone", "red", "dark chrome"
]

    PROTECTED_KEYWORDS = [
        "airbag", "cluster", "pedal", "upholstery", "seat", "safety"
    ]

    for kw in COSMETIC_ONLY_KEYWORDS:
        if kw in text and not any(pk in text for pk in PROTECTED_KEYWORDS):
            return "IGNORE", None


    # 2️⃣ MERGE via regex rules
    for pattern, master in MERGE_RULES.items():
        if re.search(pattern, text):
            return "MERGE", master

    # 3️⃣ Already mapped (CRITICAL FIX)
    if pd.notna(mapped_master) and str(mapped_master).strip():
        return "MAPPED", str(mapped_master).strip()

    # 4️⃣ Likely NEW feature
    for hint in NEW_FEATURE_HINTS:
        if hint in text:
            return "NEW", None

    # 5️⃣ Needs LLM judgment
    return "REVIEW_LLM", None


# ============================================================
# LLM HELPERS
# ============================================================

def build_llm_prompt(pending, feature_master):
    return f"""
You are an automotive feature architect.

Decide action:
- MAP → existing master
- MERGE → existing master
- NEW → create new master
- IGNORE → cosmetic / duplicate / marketing

Rules:
- DO NOT invent names
- DO NOT change category
- Prefer existing masters
- Respond STRICT JSON

Input:
{json.dumps(pending, indent=2)}

Master Features:
{json.dumps(feature_master, indent=2)}

Output:
[
  {{
    "original_name": "...",
    "action": "MAP | MERGE | NEW | IGNORE",
    "master_feature": "feature name or null"
  }}
]
"""


def call_llm(prompt: str) -> list:
    logger.info("Calling Gemini for review resolution")

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

    return json.loads(text)


# ============================================================
# CORE PROCESSOR
# ============================================================

def process_review_excel(
    excel_path: str,
    feature_master_json: str,
    output_excel: str
):
    start = time.time()
    logger.info("Reading Excel: %s", excel_path)

    df = pd.read_excel(excel_path)
    feature_master = json.load(open(feature_master_json, "r", encoding="utf-8"))

    df["final_action"] = None
    df["final_master_feature"] = None

    llm_pending = []

    # -------- RULE PASS --------
    for idx, row in df.iterrows():
        if row["mapping_status"] != "REVIEW":
            df.at[idx, "final_action"] = row["mapping_status"]
            df.at[idx, "final_master_feature"] = row["master_feature_name"]
            continue

        action, master = auto_decide(
            row["original_name"],
            row.get("master_feature_name")
        )

        df.at[idx, "final_action"] = action
        df.at[idx, "final_master_feature"] = master

        if action == "REVIEW_LLM":
            llm_pending.append({
                "original_name": row["original_name"],
                "category": row.get("Category")
            })

    logger.info("LLM needed for %d rows", len(llm_pending))

    # -------- LLM PASS --------
    if llm_pending:
        prompt = build_llm_prompt(llm_pending, feature_master)
        llm_results = call_llm(prompt)

        for res in llm_results:
            mask = df["original_name"] == res["original_name"]
            df.loc[mask, "final_action"] = res["action"]
            df.loc[mask, "final_master_feature"] = res["master_feature"]

    # -------- DB READY FILTER --------
    # db_df = df[df["final_action"].isin(["MAPPED", "MERGE", "NEW"])].copy()

    # db_df.to_excel(output_excel, index=False)
    df.to_excel(output_excel, index=False)


    logger.info("FINAL DB Excel written: %s", output_excel)
    logger.info("TOTAL TIME: %.2fs", time.time() - start)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    process_review_excel(
        excel_path="creta_data_final.xlsx",
        feature_master_json="feature_master.json",
        output_excel="creta_data_final_db2.xlsx"
    )
