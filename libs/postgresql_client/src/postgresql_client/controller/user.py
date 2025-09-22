from __future__ import annotations

from abc import ABC
from logger import get_logger
from collections.abc import Sequence
from sqlalchemy.orm import Session, joinedload
from uuid import UUID


from ..model import UserModel, UserRoleModel, RoleModel, RolePermissionModel

logger = get_logger(__name__)


# --- User Controller ---
class UserController(ABC):
    def get_users(
        self,
        session: Session,
        filter: dict[str, object] | None = None,
        order_by: Sequence | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[UserModel] | None:
        try:
            statement = session.query(UserModel).options(
                joinedload(UserModel.user_roles)
                .joinedload(UserRoleModel.role)
                .joinedload(RoleModel.role_permissions)
                .joinedload(RolePermissionModel.permission)
            )
            if filter:
                statement = statement.filter_by(**filter)
            if order_by:
                statement = statement.order_by(*order_by)
            if offset:
                statement = statement.offset(offset)
            if limit:
                statement = statement.limit(limit)

            users = statement.all()
            if not users:
                return None

            return users
        except Exception as e:
            logger.exception(f"Error fetching users: {e}", filter=filter, limit=limit)
            raise e

    def get_user_by_id(self, session: Session, id: str) -> UserModel:
        try:
            user = (
                session.query(UserModel)
                .options(
                    joinedload(UserModel.user_roles)
                    .joinedload(UserRoleModel.role)
                    .joinedload(RoleModel.role_permissions)
                    .joinedload(RolePermissionModel.permission)
                )
                .filter(UserModel.id == id)
                .one_or_none()
            )
            if not user:
                logger.info(f"No User found with id: {id}")
                return None

            return user
        except Exception as e:
            logger.exception(f"Error fetching user by id: {e}", id=id)
            raise e
    
    def get_user_by_username(
        self, session: Session, username: str
    ) -> UserModel:
        try:
            user = (
                session.query(UserModel)
                .filter(UserModel.username == username)
                .one_or_none()
            )
            if not user:
                logger.info(f"No User found with id: {id}")
                return None

            return user
        except Exception as e:
            logger.exception(f"Error fetching user by id: {e}", id=id)
            raise e
        
    def get_user_by_email(
        self, session: Session, email: str
    ) -> UserModel:
        try:
            user = (
                session.query(UserModel)
                .filter(UserModel.email == email)
                .one_or_none()
            )
            if not user:
                logger.info(f"No User found with id: {id}")
                return None

            return user
        except Exception as e:
            logger.exception(f"Error fetching user by id: {e}", id=id)
            raise e

    def insert_user(
        self, session: Session, data: UserModel, role_ids: list[str]
    ) -> UserModel:
        try:
            user_obj = data
            session.add(user_obj)
            session.flush()

            new_roles = [
                UserRoleModel(user_id=user_obj.id, role_id=UUID(role_id))
                for role_id in role_ids
            ]
            user_obj.user_roles = new_roles

            session.commit()
            session.refresh(user_obj)

            return self.get_user_by_id(session, str(user_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f"Error inserting user: {e}", data=data, role_ids=role_ids)
            raise e

    def update_user(
        self, session: Session, data: UserModel, role_ids: list[str] | None = None
    ) -> UserModel:
        try:
            user_obj = session.get(UserModel, data.id)
            if not user_obj:
                logger.info(f"No User found with id: {data.id}")
                return None

            for key, value in data.model_dump(
                exclude_none=True, exclude={"id", "roles"}
            ).items():
                if value is not None:
                    setattr(user_obj, key, value)

            if role_ids:
                user_obj.user_roles.clear()

                new_roles = [
                    UserRoleModel(user_id=UUID(data.id), role_id=UUID(role_id))
                    for role_id in role_ids
                ]
                user_obj.user_roles = new_roles

            session.commit()
            session.refresh(user_obj)

            return self.get_user_by_id(session, str(user_obj.id))

        except Exception as e:
            session.rollback()
            logger.exception(f"Error updating user: {e}", data=data)
            raise e

    def delete_user(self, session: Session, id: str) -> UserModel | None:
        try:
            user_obj = session.get(UserModel, id)
            if user_obj:
                session.delete(user_obj)
                session.commit()
                return user_obj
            else:
                logger.info(f"No User found with id: {id}")
                return None
        except Exception as e:
            session.rollback()
            logger.exception(f"Error deleting user: {e}", id=id)
            raise e
