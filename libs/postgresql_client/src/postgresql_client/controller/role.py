from __future__ import annotations

from abc import ABC
from uuid import UUID
from logger import get_logger
from collections.abc import Sequence
from sqlalchemy.orm import Session, joinedload

from ..model import (
    RoleModel,
    RolePermissionModel,
)

logger = get_logger(__name__)


class RoleController(ABC):
    def get_role(
        self,
        session: Session,
        filter: dict[str, object] | None = None,
        order_by: Sequence | None = None,
        limit: int | None = None,
    ) -> list[RoleModel] | None:
        try:
            statement = session.query(RoleModel).options(
                joinedload(RoleModel.role_permissions).joinedload(
                    RolePermissionModel.permission
                )
            )
            if filter:
                statement = statement.filter_by(**filter)
            if order_by:
                statement = statement.order_by(*order_by)
            if limit:
                statement = statement.limit(limit)

            roles = statement.all()
            if not roles:
                return None

            return roles

        except Exception as e:
            logger.exception(f"Error fetching roles: {e}", filter=filter, limit=limit)
            raise e

    def get_role_by_id(self, session: Session, id: str) -> RoleModel | None:
        try:
            role = (
                session.query(RoleModel)
                .options(
                    joinedload(RoleModel.role_permissions).joinedload(
                        RolePermissionModel.permission
                    )
                )
                .filter(RoleModel.id == id)
                .one_or_none()
            )
            if not role:
                logger.info(f"No Role found with id: {id}")
                return None

            return role
        except Exception as e:
            logger.exception(f"Error fetching role by id: {e}", id=id)
            raise e
        
    def get_role_by_name(self, session: Session, name: str) -> RoleModel | None:
        try:
            role = (
                session.query(RoleModel)
                .options(
                    joinedload(RoleModel.role_permissions).joinedload(
                        RolePermissionModel.permission
                    )
                )
                .filter(RoleModel.name == name)
                .one_or_none()
            )
            if not role:
                logger.info(f"No Role found with name: {name}")
                return None

            return role
        except Exception as e:
            logger.exception(f"Error fetching role by id: {e}", name=name)
            raise e

    def insert_role(
        self, session: Session, data: RoleModel, permission_ids: list[str]
    ) -> RoleModel:
        try:
            role_obj = data
            session.add(role_obj)
            session.flush()

            new_permissions = [
                RolePermissionModel(role_id=data.id, permission_id=UUID(perm_id))
                for perm_id in permission_ids
            ]
            role_obj.role_permissions = new_permissions

            session.commit()
            session.refresh(role_obj)

            return self.get_role_by_id(session, str(role_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(
                f"Error inserting role: {e}", data=data, permission_ids=permission_ids
            )
            raise e

    def update_role(
        self, session: Session, data: RoleModel, permission_ids: list[str]
    ) -> RoleModel | None:
        try:
            role_obj = (
                session.query(RoleModel)
                .options(
                    joinedload(RoleModel.role_permissions).joinedload(
                        RolePermissionModel.permission
                    )
                )
                .filter(RoleModel.id == data.id)
                .one_or_none()
            )
            if not role_obj:
                logger.info(f"No Role found with id: {data.id}")
                return None

            if data.name:
                role_obj.name = data.name

            role_obj.role_permissions.clear()

            new_permissions = [
                RolePermissionModel(role_id=UUID(data.id), permission_id=UUID(perm_id))
                for perm_id in permission_ids
            ]
            role_obj.role_permissions = new_permissions

            session.commit()
            return self.get_role_by_id(session, str(role_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f"Error updating role: {e}", data=data)
            raise e

    def delete_role(self, session: Session, id: str) -> RoleModel | None:
        try:
            role_obj = session.get(RoleModel, id)
            if role_obj:
                session.delete(role_obj)
                session.commit()
                return RoleModel(name=role_obj.name)
            else:
                logger.info(f"No Role found with id: {id}")
                return None
        except Exception as e:
            session.rollback()
            logger.exception(f"Error deleting role: {e}", id=id)
            raise e
