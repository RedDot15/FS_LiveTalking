from __future__ import annotations

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any
from pydantic import Field
from pydantic import ConfigDict

from logger import get_logger

from mongo_client.controller.request import RequestHandler

logger = get_logger(__name__)

class GetRequestCharacterInput(BaseModel):
    pass 

class GetRequestCharacterOutput(BaseModel):
    characters: list
    
class GetRequestCharacterApplication(BaseService):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    def process(self) -> GetRequestCharacterOutput:
        try:
            with self.request.app.state.mongodb_client.get_database() as db:
                request_handler = RequestHandler(
                    collection=db['requests']
                )
                
                results = request_handler.get_requests()
        except Exception as e:
            logger.exception('Error while get request characters', extra={'error': str(e)})
            raise e
                
        return GetRequestCharacterOutput(characters=results)

    def get_by_created_by(self, created_by: str) -> GetRequestCharacterOutput:
        try:
            with self.request.app.state.mongodb_client.get_database() as db:
                request_handler = RequestHandler(
                    collection=db['requests']
                )
                
                results = request_handler.get_requests_by_created_by(created_by=created_by)
        except Exception as e:
            logger.exception('Error while get request characters', extra={'error': str(e)})
            raise e
                
        return GetRequestCharacterOutput(characters=results)