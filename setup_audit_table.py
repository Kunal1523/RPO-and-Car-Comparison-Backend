from DBManager import DbManager

sql = """
CREATE TABLE IF NOT EXISTS master_audit_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    performed_by    TEXT NOT NULL DEFAULT 'admin',
    section         TEXT NOT NULL,
    action          TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT,
    entity_name     TEXT,
    old_value       TEXT,
    new_value       TEXT,
    meta            JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_section   ON master_audit_logs (section);
CREATE INDEX IF NOT EXISTS idx_audit_action    ON master_audit_logs (action);
CREATE INDEX IF NOT EXISTS idx_audit_entity    ON master_audit_logs (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_created   ON master_audit_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_performed ON master_audit_logs (performed_by);
"""

def run():
    db = DbManager()
    conn = db.get_conn()
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    print("Table created successfully")

if __name__ == '__main__':
    run()
