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
    def get_roles(
        self,
        session: Session,
        filter: dict[str, object] | None = None,
        order_by: Sequence | None = None,
        offset: int | None = None,
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
            if offset:
                statement = statement.offset(offset)
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
        self, session: Session, db_obj: RoleModel, permission_ids: list[str]
    ) -> RoleModel:
        try:
            session.add(db_obj)
            session.flush()

            new_permissions = [
                RolePermissionModel(role_id=db_obj.id, permission_id=UUID(perm_id))
                for perm_id in permission_ids
            ]
            db_obj.role_permissions = new_permissions

            session.commit()
            session.refresh(db_obj)

            return self.get_role_by_id(session, str(db_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(
                f"Error inserting role: {e}", data=db_obj, permission_ids=permission_ids
            )
            raise e

    def update_role(
        self, session: Session, db_obj: RoleModel, permission_ids: list[str]
    ) -> RoleModel | None:
        try:
            if permission_ids:
                db_obj.role_permissions.clear()
                new_permissions = [
                    RolePermissionModel(role_id=db_obj.id, permission_id=UUID(perm_id))
                    for perm_id in permission_ids
                ]
                db_obj.role_permissions = new_permissions

            session.commit()
            return db_obj

        except Exception as e:
            session.rollback()
            logger.exception(f"Error updating role: {e}", data=db_obj)
            raise e

    def delete_role(self, session: Session, id: str) -> RoleModel | None:
        try:
            role_obj = session.get(RoleModel, id)
            if role_obj:
                if role_obj.name == "ADMIN":
                    raise Exception("Cannot delete Role: ADMIN.")

                session.delete(role_obj)
                session.commit()
                return RoleModel(name=role_obj.name)
            else:
                logger.info(f"No Role found with id: {id}")
                return None
        except IntegrityError as ie:
            session.rollback()
            logger.exception("Cannot delete Role: It is currently assigned to users.")
            raise Exception("Cannot delete Role: It is currently assigned to users.")
        except Exception as e:
            session.rollback()
            logger.exception(f"Error deleting role: {e}", id=id)
            raise e
