from __future__ import annotations

from abc import ABC
from logger import get_logger
from collections.abc import Sequence
from sqlalchemy.orm import Session

from functools import partial

from ..model import PermissionModel

from .utils import _get_data, _get_data_by_id, _insert, _update, _delete

logger = get_logger(__name__)

_get_permission_method = partial(_get_data, logger, PermissionModel)
_get_permission_by_id_method = partial(_get_data_by_id, logger, PermissionModel)
_insert_permission_method = partial(_insert, logger, PermissionModel)
_update_permission_method = partial(_update, logger, PermissionModel)
_delete_permission_method = partial(_delete, logger, PermissionModel)


class PermissionController(ABC):
    def get_permission(
        self,
        session: Session,
        filter: dict[str, object] | None = None,
        order_by: Sequence | None = None,
        limit: int | None = None,
    ) -> list[PermissionModel] | None:
        results = _get_permission_method(session, filter, order_by, limit)
        return results if results else None

    def get_permission_by_id(self, session: Session, id: str) -> PermissionModel | None:
        result = _get_permission_by_id_method(session, id)
        return result if result else None

    def insert_permission(
        self, session: Session, data: PermissionModel
    ) -> PermissionModel:
        return _insert_permission_method(session, data)

    def update_permission(
        self, session: Session, data: PermissionModel
    ) -> PermissionModel | None:
        return _update_permission_method(session, data)

    def delete_permission(self, session: Session, id: str) -> PermissionModel | None:
        result = _delete_permission_method(session, id)
        return result if result else None
