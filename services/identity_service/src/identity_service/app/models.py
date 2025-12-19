import uuid

from pydantic import EmailStr
from sqlmodel import Field, SQLModel

######## Admin API ########
class UserCreate(SQLModel):
    username: str
    password: str = Field(min_length=8, max_length=40)
    name: str
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    phone_number: str
    role_ids: list[str] = []

class UserUpdate(SQLModel):
    password: str
    name: str
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore
    phone_number: str
    role_ids: list[str] = []

class RoleCreate(SQLModel):
    name: str
    permission_ids: list[str] = []

class RoleUpdate(SQLModel):
    name: str
    permission_ids: list[str] = []
######## End Admin API ########

######## User API ########
class UserRegister(SQLModel):
    username: str
    password: str = Field(min_length=8, max_length=40)
    name: str
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    phone_number: str

class UserUpdateMe(SQLModel):
    name: str
    email: EmailStr | None = Field(unique=True, index=True, max_length=255)
    phone_number: str

class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=40)
    new_password: str = Field(min_length=8, max_length=40)
######## End User API ########


######## Response ########
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
            for rp in role_db.role_permissions
        ]
        return cls(id=role_db.id, name=role_db.name, permissions=permissions)

class UserPublic(SQLModel):
    id: uuid.UUID
    username: str
    name: str
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    phone_number: str
    roles: list[RolePublic] = []

    @classmethod
    def model_validate(cls, user_db):
        roles = [
            RolePublic.model_validate(user_role.role)
            for user_role in user_db.user_roles
        ]
        return cls(id=user_db.id, username=user_db.username, name=user_db.name, email=user_db.email, phone_number=user_db.phone_number, roles=roles)

class UsersPublic(SQLModel):
    data: list[UserPublic]

class RolesPublic(SQLModel):
    data: list[RolePublic]

# Generic message
class Message(SQLModel):
    message: str

# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"

class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=40)
