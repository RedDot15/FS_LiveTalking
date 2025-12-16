from __future__ import annotations

from .character_router import character_router
from .request_router import request_router
from .get_character_request_router import get_character_request_router
from .delete_datas_router import delete_datas_router

__all__ = [
    'character_router',
    'request_router',
    'get_character_request_router',
    'delete_datas_router'
]