"""
chatbot/db_context.py

Fetches structured context from the car comparison database to be injected
into the Gemini AI prompt, giving the model real-time knowledge of the DB.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def _get_conn():
    """Open a fresh psycopg2 connection using env vars."""
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def get_all_brands() -> list[dict]:
    """Return [{id, name}] for every brand."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, name FROM brands ORDER BY name;")
            return [dict(r) for r in cur.fetchall()]


def get_all_cars() -> list[dict]:
    """Return [{car_id, car_name, brand_name, launch_year}]."""
    sql = """
        SELECT c.id AS car_id, c.name AS car_name,
               b.name AS brand_name, c.launch_year
        FROM cars c
        JOIN brands b ON b.id = c.brand_id
        ORDER BY b.name, c.name;
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


def get_variants_with_pricing() -> list[dict]:
    """Return latest variants with their pricing details."""
    sql = """
        SELECT
            b.name  AS brand_name,
            c.name  AS car_name,
            v.id    AS variant_id,
            v.name  AS variant_name,
            v.variant_class,
            p.ex_showroom_price,
            p.currency,
            p.fuel_type,
            p.engine_type,
            p.transmission_type,
            p.paint_type,
            p.edition
        FROM variants v
        JOIN cars c ON c.id = v.car_id
        JOIN brands b ON b.id = c.brand_id
        LEFT JOIN pricing p
            ON p.variant_id = v.id AND p.is_latest = true
        WHERE v.is_latest = true
        ORDER BY b.name, c.name, v.name, p.fuel_type;
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_feature_summary() -> list[dict]:
    """Return [{category, feature_count}] as a high-level feature overview."""
    sql = """
        SELECT category, COUNT(*) AS feature_count
        FROM features_master
        WHERE is_active = true
        GROUP BY category
        ORDER BY category;
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


def get_variant_features(variant_ids: list[str]) -> list[dict]:
    """
    Fetch features for specific variant_ids (latest only).
    variant_ids: list of UUID strings.
    """
    if not variant_ids:
        return []
    sql = """
        SELECT
            vf.variant_id,
            fm.category,
            fm.name  AS feature_name,
            vf.value
        FROM variant_features vf
        JOIN features_master fm ON fm.id = vf.feature_id
        WHERE vf.variant_id = ANY(%s)
          AND vf.is_latest = true
        ORDER BY vf.variant_id, fm.category, fm.name;
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (variant_ids,))
            return [dict(r) for r in cur.fetchall()]


def search_variants_by_name(query: str) -> list[dict]:
    """Case-insensitive search for variants/cars matching a keyword."""
    sql = """
        SELECT
            b.name AS brand_name,
            c.name AS car_name,
            v.id   AS variant_id,
            v.name AS variant_name,
            v.variant_class
        FROM variants v
        JOIN cars c ON c.id = v.car_id
        JOIN brands b ON b.id = c.brand_id
        WHERE v.is_latest = true
          AND (
                LOWER(v.name) LIKE LOWER(%s)
             OR LOWER(c.name) LIKE LOWER(%s)
             OR LOWER(b.name) LIKE LOWER(%s)
          )
        ORDER BY b.name, c.name, v.name
        LIMIT 30;
    """
    pattern = f"%{query}%"
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (pattern, pattern, pattern))
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_db_context_for_prompt() -> str:
    """
    Build a compact, human-readable summary of the current DB state.
    This is injected into the Gemini system prompt so the model knows
    what data is available.
    """
    lines: list[str] = []

    # ---- Brands ----
    try:
        brands = get_all_brands()
        lines.append(f"=== BRANDS ({len(brands)} total) ===")
        lines.append(", ".join(b["name"] for b in brands) or "None")
    except Exception as e:
        lines.append(f"=== BRANDS === [error: {e}]")

    # ---- Cars ----
    try:
        cars = get_all_cars()
        lines.append(f"\n=== CARS ({len(cars)} total) ===")
        for c in cars:
            yr = f" ({c['launch_year']})" if c.get("launch_year") else ""
            lines.append(f"  • {c['brand_name']} – {c['car_name']}{yr}")
    except Exception as e:
        lines.append(f"\n=== CARS === [error: {e}]")

    # ---- Variants + Pricing ----
    try:
        vp = get_variants_with_pricing()
        lines.append(f"\n=== VARIANTS WITH PRICING ({len(vp)} rows) ===")
        # Group by brand / car for readability
        groups: dict[str, list] = {}
        for row in vp:
            key = f"{row['brand_name']} – {row['car_name']}"
            groups.setdefault(key, []).append(row)

        for car_label, rows in groups.items():
            lines.append(f"\n  {car_label}:")
            for r in rows:
                price_str = (
                    f"₹{r['ex_showroom_price']:,.0f} {r['currency']}"
                    if r["ex_showroom_price"]
                    else "price N/A"
                )
                fuel = r.get("fuel_type") or ""
                trans = r.get("transmission_type") or ""
                paint = r.get("paint_type") or ""
                config = ", ".join(filter(None, [fuel, trans, paint]))
                config_str = f" [{config}]" if config else ""
                lines.append(
                    f"    - {r['variant_name']}{config_str}: {price_str}"
                )
    except Exception as e:
        lines.append(f"\n=== VARIANTS === [error: {e}]")

    # ---- Feature Categories ----
    try:
        feat_summary = get_feature_summary()
        lines.append(f"\n=== FEATURE CATEGORIES ({len(feat_summary)}) ===")
        for fs in feat_summary:
            lines.append(f"  • {fs['category']}: {fs['feature_count']} features")
    except Exception as e:
        lines.append(f"\n=== FEATURES === [error: {e}]")

    return "\n".join(lines)
