from __future__ import annotations

from fastapi import APIRouter

from .v1 import character_router

character_management_router = APIRouter(prefix='/v1')

character_management_router.include_router(character_router, tags=['Character Management'])
