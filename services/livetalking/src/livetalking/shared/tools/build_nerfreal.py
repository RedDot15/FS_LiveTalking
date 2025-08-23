from __future__ import annotations

from ..utils import get_settings
from realistic import LipReal

from typing import Any
import asyncio

settings = get_settings()

def nerfreal(avatar: tuple[list, list, Any], model: Any, sessionid: str) -> LipReal:
    
    lipreal = LipReal(
        avatar=avatar,
        model=model,
        sessionid=sessionid,
        settings=settings.lipreal
    )
    return lipreal

async def build_nerfreal(nerfreals: list, avatar: tuple[list, list, Any], model: Any, sessionid: id) -> LipReal:
    
    lipreal = await asyncio.get_event_loop().run_in_executor(
        None, nerfreal, avatar, model, sessionid
    )
    
    nerfreals[sessionid] = lipreal
    return lipreal
    