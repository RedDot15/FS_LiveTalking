from __future__ import annotations

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from realistic import LipReal

class OfferApplicationInput(BaseModel):
    sdp: str
    type: str
    
class OfferApplicationOutput(BaseModel):
    sdp: str
    type: str
    sessionid: int

class OfferApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    