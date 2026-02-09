from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# --- Shared Schemas ---
class GridLayout(BaseModel):
    colWidths: Dict[str, float] = {}
    rowHeights: Dict[str, float] = {}

class PlanData(BaseModel):
    regulationCells: Dict[str, List[str]]
    regOrder: Optional[List[str]] = None
    customModels: List[str] = []
    customRegulations: List[str] = []
    layout: Optional[GridLayout] = None

# --- Draft Schemas ---
class DraftBase(BaseModel):
    id: str
    name: str
    ownerEmail: str = Field(alias="owner_email")
    updatedAt: int = Field(alias="updated_at") 
    data: PlanData

    class Config:
        populate_by_name = True
        from_attributes = True

class DraftCreate(BaseModel):
    id: str
    name: str
    updatedAt: int
    data: PlanData

# --- Final Plan Schemas ---
class FinalPlanBase(BaseModel):
    ownerEmail: str = Field(alias="owner_email")
    publishedAt: int = Field(alias="published_at")
    publishedBy: Optional[str] = Field(alias="published_by", default=None)
    data: PlanData

    class Config:
        populate_by_name = True
        from_attributes = True

class FinalPlanCreate(BaseModel):
    publishedAt: int
    data: PlanData

class FinalPlanOut(FinalPlanBase):
    missingByReg: Dict[str, List[str]] = {}

# --- Regulation Schemas ---
class RegulationItem(BaseModel):
    name: str

    class Config:
        from_attributes = True

# --- Tenant User Schemas ---
class TenantUserBase(BaseModel):
    email: str
    name: str
    
    class Config:
        from_attributes = True

class ModelItemCreate(BaseModel):
    name: str

class ModelItemOut(BaseModel):
    name: str
    class Config:
        from_attributes = True

class UserSettingsOut(BaseModel):
    ownerEmail: str = Field(alias="owner_email")
    yearsWindow: int = Field(alias="years_window")
    startMonth: int = Field(alias="start_month")
    preferences: Dict[str, Any] = {}

    class Config:
        populate_by_name = True
        from_attributes = True

class UserSettingsUpdate(BaseModel):
    yearsWindow: Optional[int] = None
    startMonth: Optional[int] = None
    preferences: Optional[Dict[str, Any]] = None

class AuditLogOut(BaseModel):
    action: str
    createdAt: int = Field(alias="created_at")
    details: Dict[str, Any] = {}

    class Config:
        populate_by_name = True
        from_attributes = True