import json
from typing import Optional, Any
from datetime import datetime

audit_log_code = """

# ─────────────────────────────────────────────────────────────────────────────
# DB Manager
# ─────────────────────────────────────────────────────────────────────────────
import json
from typing import Optional, Any
from datetime import datetime
from DBManager import DbManager

class AuditLogDbManager(DbManager):
    \"\"\"Write + read audit log entries for MasterPage changes.\"\"\"

    def log(
        self,
        section: str,
        action: str,
        entity_type: str,
        entity_name: str,
        performed_by: str = "admin",
        entity_id: Optional[str] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        meta: Optional[dict] = None,
    ) -> dict:
        \"\"\"Insert one audit log row. Returns the created row.\"\"\"
        def _ser(v):
            if v is None:
                return None
            if isinstance(v, (dict, list)):
                return json.dumps(v)
            return str(v)

        query = \"\"\"
            INSERT INTO master_audit_logs
                (performed_by, section, action, entity_type, entity_id,
                 entity_name, old_value, new_value, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING id, performed_by, section, action, entity_type,
                      entity_id, entity_name, old_value, new_value, meta, created_at;
        \"\"\"
        with self.get_conn().cursor() as cur:
            cur.execute(query, (
                performed_by, section, action, entity_type, entity_id,
                entity_name, _ser(old_value), _ser(new_value),
                json.dumps(meta) if meta else None,
            ))
            r = cur.fetchone()
            self.get_conn().commit()

        return self._row_to_dict(r)

    def get_logs(
        self,
        section: Optional[str] = None,
        action: Optional[str] = None,
        entity_type: Optional[str] = None,
        performed_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        \"\"\"Fetch audit logs with optional filters. Returns {logs, total}.\"\"\"
        filters = []
        params = []

        if section:
            filters.append("section = %s")
            params.append(section)
        if action:
            filters.append("action = %s")
            params.append(action)
        if entity_type:
            filters.append("entity_type = %s")
            params.append(entity_type)
        if performed_by:
            filters.append("performed_by = %s")
            params.append(performed_by)

        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        query = f\"\"\"
            SELECT id, performed_by, section, action, entity_type,
                   entity_id, entity_name, old_value, new_value, meta, created_at
            FROM master_audit_logs
            {where}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s;
        \"\"\"
        count_query = f"SELECT COUNT(*) FROM master_audit_logs {where};"

        with self.get_conn().cursor() as cur:
            cur.execute(count_query, params)
            total = cur.fetchone()[0]
            cur.execute(query, params + [limit, offset])
            rows = cur.fetchall()

        return {
            "logs": [self._row_to_dict(r) for r in rows],
            "total": total,
        }

    @staticmethod
    def _row_to_dict(r) -> dict:
        return {
            "id":           str(r[0]),
            "performed_by": r[1],
            "section":      r[2],
            "action":       r[3],
            "entity_type":  r[4],
            "entity_id":    str(r[5]) if r[5] else None,
            "entity_name":  r[6],
            "old_value":    r[7],
            "new_value":    r[8],
            "meta":         r[9] or {},
            "created_at":   r[10].isoformat() if isinstance(r[10], datetime) else str(r[10]),
        }
"""

with open("d:/RPO-CAR-BACKEND/RPO and Car Comparison/DBManager.py", "a", encoding="utf-8") as f:
    f.write(audit_log_code)
print("Appended successfully.")
