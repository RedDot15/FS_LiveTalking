from __future__ import annotations

from base import CustomBaseModel

from livetalking.shared.models import RecordType

class AudioTypeApplicationInput(CustomBaseModel):
    sessionid: int | None = None
    audiotype: str | None = None
    reinit: bool | None = None
    
class RecordApplicationInput(CustomBaseModel):
    sessionid: int
    type: RecordType
    
class IsSpeakingApplicationInput(CustomBaseModel):
    sessionid: int
    