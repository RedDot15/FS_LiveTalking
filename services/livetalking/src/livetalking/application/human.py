from __future__ import annotations

from base import BaseModel

from livetalking.shared.models import HumanType

class HumanApplicationInput(BaseModel):
    character_id: str
    type: HumanType
    text: str
    interrupt: bool | None = None

