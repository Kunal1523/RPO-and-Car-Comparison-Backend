from sqlalchemy import Column, String, Integer, BigInteger, JSON, Boolean
from database import Base

class Draft(Base):
    __tablename__ = "drafts"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    owner_email = Column(String, index=True)
    updated_at = Column(BigInteger)
    data = Column(JSON)

class FinalPlan(Base):
    __tablename__ = "final_plans"
    # per-user final plan
    owner_email = Column(String, primary_key=True, index=True)
    published_at = Column(BigInteger)
    published_by = Column(String)
    data = Column(JSON)

class Regulation(Base):
    __tablename__ = "regulations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_email = Column(String, index=True)
    name = Column(String, index=True)
    is_archived = Column(Boolean, default=False)

class ModelItem(Base):
    """
    User-maintained list of models used for compliance tracking.
    """
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_email = Column(String, index=True)
    name = Column(String, index=True)
    is_archived = Column(Boolean, default=False)

class TenantUser(Base):
    __tablename__ = "tenant_users"
    email = Column(String, primary_key=True, index=True)
    name = Column(String)
    tenant_id = Column(String, index=True)
    is_active = Column(Integer, default=1)

class UserSettings(Base):
    """
    Per-user preferences (years window etc.)
    """
    __tablename__ = "user_settings"
    owner_email = Column(String, primary_key=True, index=True)
    years_window = Column(Integer, default=3)      # e.g. 3 years shown
    start_month = Column(Integer, default=4)       # FY start month (April)
    preferences = Column(JSON, default={})         # any extra UI prefs

class AuditLog(Base):
    """
    Track publish/share actions with timestamps.
    """
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_email = Column(String, index=True)       # who did the action
    action = Column(String, index=True)            # PUBLISH_FINAL, SHARE_FINAL, etc
    created_at = Column(BigInteger, index=True)    # Date.now() from backend
    details = Column(JSON, default={})             # recipients, subject, counts, etc
