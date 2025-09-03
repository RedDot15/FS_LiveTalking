from __future__ import annotations

from abc import ABC
from typing import cast
from logger import get_logger
from collections.abc import Sequence
from sqlalchemy.orm import Session, joinedload
from uuid import UUID, uuid4

from functools import partial

from ..model import (
    UserModel,
    UserRoleModel,
    RoleModel,
    RolePermissionModel,
    User,
    Role,
    Permission
)

from .utils import (
    _get_data,
    _get_data_by_id,
    _insert,
    _update,
    _delete
)

logger = get_logger(__name__)

# --- User Controller ---

_get_method = partial(_get_data, logger, UserModel, User)
_get_data_by_id_method = partial(_get_data_by_id, logger, UserModel, User)
_insert_method = partial(_insert, logger, UserModel, User)
_update_method = partial(_update, logger, UserModel, User)
_delete_method = partial(_delete, logger, UserModel, User)

class UserController(ABC):

    def get_user(self, 
                 session: Session, 
                 filter: dict[str, object] | None = None, 
                 order_by: Sequence | None = None,
                 limit: int | None = None) -> list[User] | None:
        try:
            statement = session.query(UserModel).options(joinedload(UserModel.userRoles)
                                                         .joinedload(UserRoleModel.role)
                                                         .joinedload(RoleModel.rolePermissions)
                                                         .joinedload(RolePermissionModel.permission))
            if filter:
                statement = statement.filter_by(**filter)
            if order_by:
                statement = statement.order_by(*order_by)
            if limit:
                statement = statement.limit(limit)
            
            users = statement.all()
            if not users:
                return None
            
            return [
                User(
                    username=user.username,
                    password=user.password,
                    name=user.name,
                    avatar_url=user.avatar_url,
                    email=user.email,
                    phone_number=user.phone_number,
                    roles=[
                        Role(
                            name=ur.role.name,
                            permissions=[
                                Permission(name=rp.permission.name) for rp in ur.role.rolePermissions  
                            ]
                        ) for ur in user.userRoles
                    ]
                ) for user in users
            ]
        except Exception as e:
            logger.exception(f'Error fetching users: {e}', filter=filter, limit=limit)
            raise e

    def get_user_by_id(self, 
                       session: Session, 
                       id: str) -> User:
        try:
            user = session.query(UserModel).options(joinedload(UserModel.userRoles)
                                                    .joinedload(UserRoleModel.role)
                                                    .joinedload(RoleModel.rolePermissions)
                                                    .joinedload(RolePermissionModel.permission)
                                                    ).filter(UserModel.id == id).one_or_none()
            if not user:
                logger.info(f'No User found with id: {id}')
                return None

            return User(
                    username=user.username,
                    password=user.password,
                    name=user.name,
                    avatar_url=user.avatar_url,
                    email=user.email,
                    phone_number=user.phone_number,
                    roles=[
                        Role(
                            name=ur.role.name,
                            permissions=[
                                Permission(name=rp.permission.name) for rp in ur.role.rolePermissions  
                            ]
                        ) for ur in user.userRoles
                    ]
            )
        except Exception as e:
            logger.exception(f'Error fetching user by id: {e}', id=id)
            raise e

    def insert_user(self, 
                    session: Session, 
                    data: User,
                    role_ids: list[str]) -> User:
        try:
            new_uuid = uuid4()
            user_obj = UserModel(id=new_uuid, **data.model_dump(exclude={'roles'}))
            session.add(user_obj)
            session.flush()

            new_roles = [
                UserRoleModel(
                    user_id=new_uuid, 
                    role_id=UUID(role_id)
                )
                for role_id in role_ids
            ]
            user_obj.userRoles = new_roles
            
            session.commit()
            session.refresh(user_obj)
            
            return self.get_user_by_id(session, str(user_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f'Error inserting user: {e}', data=data, role_ids=role_ids)
            raise e
    
    def update_user(self,
                    session: Session,
                    data: User,
                    id: str,
                    role_ids: list[str]) -> User:
        try:
            user_obj = session.get(UserModel, id)
            if not user_obj:
                logger.info(f'No User found with id: {id}')
                return None

            for key, value in data.model_dump(exclude_none=True, exclude={'id', 'roles'}).items():
                if value is not None:
                    setattr(user_obj, key, value)
            
            user_obj.userRoles.clear()

            new_roles = [
                UserRoleModel(
                    user_id=UUID(id), 
                    role_id=UUID(role_id)
                )
                for role_id in role_ids
            ]
            user_obj.userRoles = new_roles

            session.commit()
            session.refresh(user_obj)
            
            return self.get_user_by_id(session, str(user_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f'Error updating user: {e}', data=data)
            raise e
    
    def delete_user(self,
                    session: Session,
                    id: str) -> User | None:
        try:
            user_obj = session.get(UserModel, id)
            if user_obj:
                session.delete(user_obj)
                session.commit()
                return cast(User, user_obj)
            else:
                logger.info(f'No User found with id: {id}')
                return None
        except Exception as e:
            session.rollback()
            logger.exception(f'Error deleting user: {e}', id=id)
            raise e