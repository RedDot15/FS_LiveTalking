from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext
from postgresql_client.model.models import UserModel

from identity_service.app.core import security
from identity_service.app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


ALGORITHM = "HS256"

def build_scope(db_user: UserModel) -> str:
    permissions = []
    for user_role in db_user.user_roles:
        for role_permission in user_role.role.role_permissions:
            permissions.append(role_permission.permission.name)
    return " ".join(permissions)

def create_access_token(db_user: UserModel | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"exp": expire, "id": str(db_user.id), "scope": build_scope(db_user)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[security.ALGORITHM])
    except (InvalidTokenError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

def decode_token(token: str):
    return jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM], options={"verify_signature": False}
    )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
