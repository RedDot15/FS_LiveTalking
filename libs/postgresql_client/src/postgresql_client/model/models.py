from __future__ import annotations
import uuid

from sqlalchemy import ForeignKey, Column, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "user"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    avatar_url = Column(String)
    email = Column(String, unique=True, nullable=False)
    phone_number = Column(String)
    user_roles = relationship(
        "UserRoleModel", back_populates="user", cascade="all, delete-orphan"
    )


class RoleModel(Base):
    __tablename__ = "role"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    role_permissions = relationship(
        "RolePermissionModel", back_populates="role", cascade="all, delete-orphan"
    )
    user_roles = relationship(
        "UserRoleModel", back_populates="role", cascade="all, delete-orphan"
    )


class PermissionModel(Base):
    __tablename__ = "permission"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    role_permissions = relationship(
        "RolePermissionModel", back_populates="permission", cascade="all, delete-orphan"
    )


class UserRoleModel(Base):
    __tablename__ = "user_role"
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), primary_key=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey("role.id"), primary_key=True)
    user = relationship("UserModel", back_populates="user_roles")
    role = relationship("RoleModel", back_populates="user_roles")


class RolePermissionModel(Base):
    __tablename__ = "role_permission"
    role_id = Column(UUID(as_uuid=True), ForeignKey("role.id"), primary_key=True)
    permission_id = Column(
        UUID(as_uuid=True), ForeignKey("permission.id"), primary_key=True
    )
    role = relationship("RoleModel", back_populates="role_permissions")
    permission = relationship("PermissionModel", back_populates="role_permissions")
