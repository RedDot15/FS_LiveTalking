from __future__ import annotations
from typing import List

from pydantic import ConfigDict, BaseModel

class BaseSchema():
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class User(BaseSchema, BaseModel):
    username: str
    password: str
    name: str
    avatar_url: str
    email: str
    phone_number: str
    roles: List[Role] = []

class Role(BaseSchema, BaseModel):
    name: str
    permissions: List[Permission] = []

class Permission(BaseSchema, BaseModel):
    name: str
