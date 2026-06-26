# routers/stackup.py
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Literal
import uuid as uuid_lib
from psycopg2.extras import RealDictCursor

from dependencies import get_current_user
from DBManager import DbManager as get_db_connection  # adapt to however the project gets a connection/pool

router = APIRouter(prefix="/api/stackup", tags=["Feature Stack-Up"])

VariantRefType = Literal["production", "new_model"]


class PrefOut(BaseModel):
    id: str
    variant_ref_type: VariantRefType
    variant_id: str
    feature_id: Optional[str]
    feature_name: str
    is_hidden: bool
    display_order: int


class BulkPrefsRequest(BaseModel):
    variant_ref_type: VariantRefType
    variant_ids: list[str]


class UpsertPrefRequest(BaseModel):
    variant_ref_type: VariantRefType
    variant_id: str
    feature_id: Optional[str] = None   # must reference an existing features_master.id, or be None
    feature_name: str
    is_hidden: Optional[bool] = None
    display_order: Optional[int] = None


class ReorderRequest(BaseModel):
    variant_ref_type: VariantRefType
    variant_id: str
    ordered_feature_names: list[str] = Field(..., min_items=1)
    hidden_states: Optional[dict[str, bool]] = None


@router.get("/prefs")
def get_prefs(
    variant_ref_type: VariantRefType,
    variant_id: str,
    user_email: str = Depends(get_current_user),
):
    db = get_db_connection()
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, variant_ref_type, variant_id, feature_id, feature_name,
                       is_hidden, display_order
                FROM user_feature_stackup_prefs
                WHERE user_id = %s AND variant_ref_type = %s AND variant_id::text = %s
                ORDER BY display_order ASC
                """,
                (user_email, variant_ref_type, variant_id),
            )
            rows = cur.fetchall()
    return {"success": True, "data": [dict(r) for r in rows]}


@router.post("/prefs/bulk")
def get_prefs_bulk(
    payload: BulkPrefsRequest,
    user_email: str = Depends(get_current_user),
):
    if not payload.variant_ids:
        return {"success": True, "data": {}}

    db = get_db_connection()
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, variant_ref_type, variant_id, feature_id, feature_name,
                       is_hidden, display_order
                FROM user_feature_stackup_prefs
                WHERE user_id = %s AND variant_ref_type = %s AND variant_id::text = ANY(%s)
                ORDER BY variant_id, display_order ASC
                """,
                (user_email, payload.variant_ref_type, payload.variant_ids),
            )
            rows = cur.fetchall()

    grouped: dict[str, list[dict]] = {vid: [] for vid in payload.variant_ids}
    for r in rows:
        grouped.setdefault(str(r["variant_id"]), []).append(dict(r))

    return {"success": True, "data": grouped}


@router.patch("/prefs")
def upsert_pref(
    payload: UpsertPrefRequest,
    user_email: str = Depends(get_current_user),
):
    db = get_db_connection()

    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # feature_id, if provided, must exist in features_master — otherwise the
            # new FK constraint will reject the insert/update. Surface that as a 400
            # instead of a raw DB error.
            if payload.feature_id is not None:
                cur.execute(
                    "SELECT 1 FROM features_master WHERE id = %s",
                    (payload.feature_id,),
                )
                exists = cur.fetchone()
                if not exists:
                    raise HTTPException(status_code=400, detail="feature_id does not exist in features_master")

            # Resolve current display_order if not provided (append to end)
            if payload.display_order is None:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(display_order), -1) + 1 AS next_order
                    FROM user_feature_stackup_prefs
                    WHERE user_id = %s AND variant_ref_type = %s AND variant_id::text = %s
                    """,
                    (user_email, payload.variant_ref_type, payload.variant_id),
                )
                max_row = cur.fetchone()
                resolved_order = max_row["next_order"] if max_row else 0
            else:
                resolved_order = payload.display_order

            cur.execute(
                """
                INSERT INTO user_feature_stackup_prefs
                    (id, user_id, variant_ref_type, variant_id, feature_id, feature_name, is_hidden, display_order, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, false), %s, now())
                ON CONFLICT (user_id, variant_ref_type, variant_id, feature_name)
                DO UPDATE SET
                    is_hidden = COALESCE(EXCLUDED.is_hidden, user_feature_stackup_prefs.is_hidden),
                    display_order = CASE
                        WHEN %s THEN EXCLUDED.display_order
                        ELSE user_feature_stackup_prefs.display_order
                    END,
                    feature_id = EXCLUDED.feature_id,
                    updated_at = now()
                RETURNING id, variant_ref_type, variant_id, feature_id, feature_name, is_hidden, display_order
                """,
                (
                    str(uuid_lib.uuid4()), user_email, payload.variant_ref_type, payload.variant_id,
                    payload.feature_id, payload.feature_name, payload.is_hidden, resolved_order,
                    payload.display_order is not None,
                ),
            )
            row = cur.fetchone()
            conn.commit()

    return {"success": True, "data": dict(row) if row else {}}


@router.patch("/prefs/reorder")
def reorder_prefs(
    payload: ReorderRequest,
    user_email: str = Depends(get_current_user),
):
    db = get_db_connection()
    updated = 0
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            for idx, feature_name in enumerate(payload.ordered_feature_names):
                is_hidden = payload.hidden_states.get(feature_name) if payload.hidden_states else None
                if is_hidden is not None:
                    cur.execute(
                        """
                        INSERT INTO user_feature_stackup_prefs
                            (id, user_id, variant_ref_type, variant_id, feature_name, display_order, is_hidden, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (user_id, variant_ref_type, variant_id, feature_name)
                        DO UPDATE SET display_order = EXCLUDED.display_order, is_hidden = EXCLUDED.is_hidden, updated_at = now()
                        """,
                        (str(uuid_lib.uuid4()), user_email, payload.variant_ref_type, payload.variant_id, feature_name, idx, is_hidden),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO user_feature_stackup_prefs
                            (id, user_id, variant_ref_type, variant_id, feature_name, display_order, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (user_id, variant_ref_type, variant_id, feature_name)
                        DO UPDATE SET display_order = EXCLUDED.display_order, updated_at = now()
                        """,
                        (str(uuid_lib.uuid4()), user_email, payload.variant_ref_type, payload.variant_id, feature_name, idx),
                    )
                updated += 1
            conn.commit()
    return {"success": True, "updated": updated}


@router.delete("/prefs")
def reset_prefs(
    variant_ref_type: VariantRefType,
    variant_id: str,
    user_email: str = Depends(get_current_user),
):
    db = get_db_connection()
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM user_feature_stackup_prefs
                WHERE user_id = %s AND variant_ref_type = %s AND variant_id::text = %s
                """,
                (user_email, variant_ref_type, variant_id),
            )
            rowcount = cur.rowcount
            conn.commit()
    return {"success": True, "deleted": rowcount}