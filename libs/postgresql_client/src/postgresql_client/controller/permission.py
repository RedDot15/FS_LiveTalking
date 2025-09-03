from __future__ import annotations

from abc import ABC
from typing import cast
from logger import get_logger
from collections.abc import Sequence
from sqlalchemy.orm import Session, joinedload
from uuid import uuid4

from functools import partial

from ..model import (
    PermissionModel,
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

_get_permission_method = partial(_get_data, logger, PermissionModel, Permission)
_get_permission_by_id_method = partial(_get_data_by_id, logger, PermissionModel, Permission)
_insert_permission_method = partial(_insert, logger, PermissionModel, Permission)
_update_permission_method = partial(_update, logger, PermissionModel, Permission)
_delete_permission_method = partial(_delete, logger, PermissionModel, Permission)

class PermissionController(ABC):
    def get_permission(self,
                 session: Session,
                 filter: dict[str, object] | None = None,
                 order_by: Sequence | None = None,
                 limit: int | None = None) -> list[Permission] | None:
        results = _get_permission_method(session, filter, order_by, limit)
        return cast(list[Permission], results) if results else None

    def get_permission_by_id(self,
                       session: Session,
                       id: str) -> Permission | None:
        result = _get_permission_by_id_method(session, id)
        return cast(Permission, result) if result else None

    def insert_permission(self,
                    session: Session,
                    data: Permission) -> Permission:
        return cast(Permission, _insert_permission_method(session, data))

    def update_permission(self,
                    session: Session,
                    data: Permission,
                    id: str) -> Permission | None:
        return cast(Permission, _update_permission_method(session, data, id))

    def delete_permission(self,
                    session: Session,
                    id: str) -> Permission | None:
        result = _delete_permission_method(session, id)
        return cast(Permission, result) if result else None
