from __future__ import annotations

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any
from pydantic import Field
from pydantic import ConfigDict

from mongo_client.controller.request import RequestHandler

class GetRequestCharacterInput(BaseModel):
    pass 

class GetRequestCharacterOutput(BaseModel):
    characters: list
    
class GetRequestCharacterApplication(BaseService):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    def process(self) -> GetRequestCharacterOutput:
        
        with self.request.app.state.mongodb_client.get_database() as db:
            request_handler = RequestHandler(
                collection=db['requests']
            )
            
            results = request_handler.get_requests()
            
        return GetRequestCharacterOutput(characters=results)