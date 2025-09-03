from __future__ import annotations

from abc import ABC
from typing import cast
from logger import get_logger
from collections.abc import Sequence
from sqlalchemy.orm import Session, joinedload
from uuid import UUID, uuid4

from functools import partial

from ..model import (
    RoleModel,
    RolePermissionModel,
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

_get_role_method = partial(_get_data, logger, RoleModel, Role)
_get_role_by_id_method = partial(_get_data_by_id, logger, RoleModel, Role)
_insert_role_method = partial(_insert, logger, RoleModel, Role)
_update_role_method = partial(_update, logger, RoleModel, Role)
_delete_role_method = partial(_delete, logger, RoleModel, Role)

class RoleController(ABC):
    def get_role(self,
                 session: Session,
                 filter: dict[str, object] | None = None,
                 order_by: Sequence | None = None,
                 limit: int | None = None) -> list[Role] | None:
        try:
            statement = session.query(RoleModel).options(joinedload(RoleModel.rolePermissions).joinedload(RolePermissionModel.permission))
            if filter:
                statement = statement.filter_by(**filter)
            if order_by:
                statement = statement.order_by(*order_by)
            if limit:
                statement = statement.limit(limit)

            roles = statement.all()
            if not roles:
                return None

            return [
                Role(
                    name=role.name,
                    permissions=[Permission(name=rp.permission.name) for rp in role.rolePermissions] 
                )
                for role in roles
            ]
        except Exception as e:
            logger.exception(f'Error fetching roles: {e}', filter=filter, limit=limit)
            raise e

    def get_role_by_id(self,
                       session: Session,
                       id: str) -> Role | None:
        try:
            role = session.query(RoleModel).options(joinedload(RoleModel.rolePermissions).joinedload(RolePermissionModel.permission)).filter(RoleModel.id == id).one_or_none()
            if not role:
                logger.info(f'No Role found with id: {id}')
                return None

            return Role(
                name=role.name,
                permissions=[Permission(name=rp.permission.name) for rp in role.rolePermissions]
            )
        except Exception as e:
            logger.exception(f'Error fetching role by id: {e}', id=id)
            raise e

    def insert_role(self,
                    session: Session,
                    data: Role,
                    permission_ids: list[str]) -> Role:
        try:
            new_uuid = uuid4()
            role_obj = RoleModel(id=new_uuid, name=data.name)
            session.add(role_obj)
            session.flush()

            new_permissions = [
                RolePermissionModel(
                    role_id=new_uuid, 
                    permission_id=UUID(perm_id)
                )
                for perm_id in permission_ids
            ]
            role_obj.rolePermissions = new_permissions

            session.commit()
            session.refresh(role_obj)

            return self.get_role_by_id(session, str(role_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f'Error inserting role: {e}', data=data, permission_ids=permission_ids)
            raise e

    def update_role(self,
                    session: Session,
                    data: Role,
                    id: str,
                    permission_ids: list[str]) -> Role | None:
        try:
            role_obj = session.query(RoleModel).options(joinedload(RoleModel.rolePermissions).joinedload(RolePermissionModel.permission)).filter(RoleModel.id == id).one_or_none()
            if not role_obj:
                logger.info(f'No Role found with id: {data.id}')
                return None

            if data.name:
                role_obj.name = data.name

            role_obj.rolePermissions.clear()

            new_permissions = [
                RolePermissionModel(
                    role_id=UUID(id), 
                    permission_id=UUID(perm_id)
                )
                for perm_id in permission_ids
            ]
            role_obj.rolePermissions = new_permissions

            session.commit()
            return self.get_role_by_id(session, str(role_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f'Error updating role: {e}', data=data)
            raise e

    def delete_role(self,
                    session: Session,
                    id: str) -> Role | None:
        try:
            role_obj = session.get(RoleModel, id)
            if role_obj:
                session.delete(role_obj)
                session.commit()
                return Role(name=role_obj.name)
            else:
                logger.info(f'No Role found with id: {id}')
                return None
        except Exception as e:
            session.rollback()
            logger.exception(f'Error deleting role: {e}', id=id)
            raise e