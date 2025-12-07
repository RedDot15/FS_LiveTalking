from __future__ import annotations

from fastapi import APIRouter

from .v1 import character_router
from .v1 import request_router
from .v1 import get_character_request_router

character_management_router = APIRouter(prefix='/v1')

character_management_router.include_router(character_router, tags=['Character Management'])
character_management_router.include_router(request_router, tags=['Request Management'])
character_management_router.include_router(get_character_request_router, tags=['Request Management'])
