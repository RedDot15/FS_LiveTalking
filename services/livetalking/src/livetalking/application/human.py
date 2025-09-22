from __future__ import annotations

from base import CustomBaseModel

from livetalking.shared.models import HumanType

class HumanApplicationInput(CustomBaseModel):
    sessionid: int | None = None
    type: HumanType
    text: str
    interrupt: bool | None = None

