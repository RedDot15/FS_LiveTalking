from __future__ import annotations

from base import BaseModel

from livetalking.shared.models import HumanType

class HumanApplicationInput(BaseModel):
    sessionid: int | None = None
    type: HumanType
    text: str
    interrupt: bool | None = None

