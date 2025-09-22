import uuid

from pydantic import EmailStr
from sqlmodel import Field, SQLModel

# Shared properties

class PermissionPublic(SQLModel):
    id: uuid.UUID
    name: str

class RolePublic(SQLModel):
    id: uuid.UUID
    name: str
    permissions: list[PermissionPublic] = []

    @classmethod
    def model_validate(cls, role_db):
        permissions = [
            PermissionPublic.model_validate(rp.permission)
            for rp in role_db.rolePermissions
        ]
        return cls(permissions=permissions, **role_db.dict())

# Properties to receive via API on creation
class UserCreate(SQLModel):
    username: str
    password: str = Field(min_length=8, max_length=40)
    name: str
    avatar_url: str
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    phone_number: str
    roles_ids: list[str] = []


class UserRegister(SQLModel):
    username: str
    password: str = Field(min_length=8, max_length=40)
    name: str
    avatar_url: str
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    phone_number: str


# Properties to receive via API on update, all are optional
class UserUpdate(SQLModel):
    password: str
    name: str
    avatar_url: str
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore
    phone_number: str
    role_ids: list[str] = []


class UserUpdateMe(SQLModel):
    name: str
    avatar_url: str
    email: EmailStr | None = Field(unique=True, index=True, max_length=255)
    phone_number: str


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=40)
    new_password: str = Field(min_length=8, max_length=40)


# Properties to return via API, id is always required
class UserPublic(SQLModel):
    id: uuid.UUID
    username: str
    name: str
    avatar_url: str
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    phone_number: str
    roles: list[RolePublic] = []

    @classmethod
    def model_validate(cls, user_db):
        roles = [
            RolePublic.model_validate(user_role.role)
            for user_role in user_db.user_roles
        ]
        return cls(roles=roles, **user_db.dict())

class UsersPublic(SQLModel):
    data: list[UserPublic]

# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    id: str | None = None
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=40)
