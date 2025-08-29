from __future__ import annotations
import uuid

from sqlalchemy import (
    ForeignKey, 
    Column, 
    String
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    avatar_url = Column(String)
    email = Column(String, unique=True, nullable=False)
    phone_number = Column(String)
    userRoles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")

class Role(Base):
    __tablename__ = "role"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    rolePermissions = relationship("RolePermission", back_populates="role", cascade="all, delete-orphan")
    userRoles = relationship("UserRole", back_populates="role", cascade="all, delete-orphan")

class Permission(Base):
    __tablename__ = "permission"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    rolePermissions = relationship("RolePermission", back_populates="permission", cascade="all, delete-orphan")

class UserRole(Base):
    __tablename__ = "user_role"
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), primary_key=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey("role.id"), primary_key=True)
    user = relationship("User", back_populates="userRoles")
    role = relationship("Role", back_populates="userRoles")

class RolePermission(Base):
    __tablename__ = "role_permission"
    role_id = Column(UUID(as_uuid=True), ForeignKey("role.id"), primary_key=True)
    permission_id = Column(UUID(as_uuid=True), ForeignKey("permission.id"), primary_key=True)
    role = relationship("Role", back_populates="rolePermissions")
    permission = relationship("Permission", back_populates="rolePermissions")