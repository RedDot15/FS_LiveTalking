from __future__ import annotations

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any
from pydantic import Field
from pydantic import ConfigDict

from logger import get_logger

from mongo_client.controller.request import RequestHandler
from mongo_client.controller.character import CharacterHandler
from mongo_client.controller.rating import RatingHandler

logger = get_logger(__name__)
    
class DeleteDatasService(BaseService):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    def process(self, user_id: str) -> str:
        try:
            with self.request.app.state.mongodb_client.get_database() as db:
                # Delete all requests by user_id
                request_handler = RequestHandler(
                    collection=db['requests']
                )
                request_handler.delete_requests_by_created_by(created_by=user_id)

                # Delete all characters by user_id
                character_handler = CharacterHandler(
                    collection=db['characters']
                )
                character_handler.delete_characters_by_created_by(created_by=user_id)

                # TODO: Delete ratings by user_id
                rating_handler = RatingHandler(
                    collection=db['ratings']
                )
                rating_handler.delete_ratings_by_created_by(created_by=user_id)
        except Exception as e:
            logger.exception('Error while get request characters', extra={'error': str(e)})
            raise e
                
        return user_id