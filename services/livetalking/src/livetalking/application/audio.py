from __future__ import annotations

from base import BaseModel

from livetalking.shared.models import RecordType

class AudioTypeApplicationInput(BaseModel):
    sessionid: int | None = None
    audiotype: str | None = None
    reinit: bool | None = None
    
class RecordApplicationInput(BaseModel):
    sessionid: int
    type: RecordType
    
class IsSpeakingApplicationInput(BaseModel):
    sessionid: int
    