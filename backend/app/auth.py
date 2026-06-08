"""Authentication & RLS identity.

- A single shared **admin** account (common login).
- **Employee** logins are identified by their Employee ID, which becomes the
  row-level-security (RLS) scope: an employee can only ever see their own rows.

Tokens are JWTs carrying role + employee_id + permissions so every downstream
step can enforce access control without re-querying.
"""
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import settings
from .schemas import LoginRequest, UserInfo

bearer_scheme = HTTPBearer(auto_error=False)

# Permission catalogue (RBAC - step 12)
EMPLOYEE_PERMISSIONS = ["faq.read", "policy.read", "self.data.read", "chat.general"]
ADMIN_PERMISSIONS = [
    "faq.read", "policy.read", "self.data.read", "all.data.read",
    "chat.general", "pii.read", "audit.read",
]


def create_access_token(claims: dict) -> str:
    to_encode = claims.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_alg)


def authenticate(req: LoginRequest) -> UserInfo:
    """Validate credentials and produce the user identity."""
    if req.role == "admin":
        username = (req.username or "").strip() or settings.admin_username
        if username != settings.admin_username or req.password != settings.admin_password:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin credentials")
        return UserInfo(
            role="admin",
            username=settings.admin_username,
            permissions=ADMIN_PERMISSIONS,
        )

    if req.role == "employee":
        emp_id = (req.employee_id or "").strip()
        if not emp_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Employee ID is required")
        if not emp_id.isdigit():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Employee ID must be numeric")
        # Common employee password gate (identity is the employee_id itself).
        if req.password != settings.employee_password:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid employee password")
        return UserInfo(
            role="employee",
            employee_id=emp_id,
            permissions=EMPLOYEE_PERMISSIONS,
        )

    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown role")


def issue_token(user: UserInfo) -> str:
    return create_access_token({
        "role": user.role,
        "employee_id": user.employee_id,
        "username": user.username,
        "permissions": user.permissions,
    })


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> UserInfo:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        payload = jwt.decode(
            creds.credentials, settings.jwt_secret, algorithms=[settings.jwt_alg]
        )
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    return UserInfo(
        role=payload.get("role", "employee"),
        employee_id=payload.get("employee_id"),
        username=payload.get("username"),
        permissions=payload.get("permissions", []),
    )
