from __future__ import annotations

from fastapi import APIRouter

from .v1 import offer_router
from .v1 import human_router
from .v1 import audio

livetalking_router = APIRouter(prefix='/v1')

livetalking_router.include_router(offer_router, tags=['LiveTalking'])
livetalking_router.include_router(human_router, tags=['LiveTalking'])
livetalking_router.include_router(audio, tags=['LiveTalking'])
