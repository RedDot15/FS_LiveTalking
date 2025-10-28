from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext
from jwt.exceptions import InvalidTokenError
from postgresql_client.model.models import UserModel

from identity_service.app.core import security
from identity_service.app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TOKEN_BLACKLIST: set = set()

def build_scope(db_user: UserModel) -> str:
    permissions = []
    for user_role in db_user.user_roles:
        for role_permission in user_role.role.role_permissions:
            permissions.append(role_permission.permission.name)
    return " ".join(permissions)

def create_access_token(db_user: UserModel | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"exp": expire, "id": str(db_user.id), "scope": build_scope(db_user)}
    encoded_jwt = jwt.encode(to_encode, settings.ACCESS_TOKEN_SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """
    Verifies the token's signature and checks if it has been blacklisted.
    """
    # First, check if the token has been logged out
    if token in TOKEN_BLACKLIST:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been logged out",
        )

    try:
        # Then, attempt to decode the token
        return jwt.decode(token, settings.ACCESS_TOKEN_SECRET_KEY, algorithms=[settings.ALGORITHM])
    except (InvalidTokenError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

def logout_token(token: str):
    """
    Invalidates a token by adding it to the blacklist.
    """
    TOKEN_BLACKLIST.add(token)
    print(f"Token invalidated. Current blacklist size: {len(TOKEN_BLACKLIST)}")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
