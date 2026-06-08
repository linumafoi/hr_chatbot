"""Shared pydantic models used across the gateway and pipeline."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------- Auth ----------------
class LoginRequest(BaseModel):
    role: str = Field(..., description="'employee' or 'admin'")
    password: str
    employee_id: Optional[str] = None
    username: Optional[str] = None


class UserInfo(BaseModel):
    role: str
    employee_id: Optional[str] = None
    username: Optional[str] = None
    department: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


# ---------------- Chat ----------------
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    agent: str
    confidence: float = 0.0
    sources: List[str] = Field(default_factory=list)
    table: Optional[List[Dict[str, Any]]] = None
    chart: Optional[Dict[str, Any]] = None
    latency_ms: int = 0
    session_id: str
    debug: Optional[Dict[str, Any]] = None


# ---------------- Pipeline context ----------------
class UserContext(BaseModel):
    """Step 6 - context extraction. Carries identity + RLS scope through the graph."""
    role: str
    employee_id: Optional[str] = None
    username: Optional[str] = None
    department: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    session_id: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"
