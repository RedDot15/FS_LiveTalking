from __future__ import annotations

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from mongo_client.controller import ConversationHandler

class CharacterServiceInput(BaseModel):
    character_id: str

class CharacterServiceOutput(BaseModel):
    characters: list[dict]

class CharacterServiceApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]

    async def process(self, input: CharacterServiceInput) -> CharacterServiceOutput:
        
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            
            try:
                conv_handler = ConversationHandler(collection=mongodb["conversations"])
                characters = conv_handler.get_conversation_by_participants_hash(character_id=input.character_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
            
        return CharacterServiceOutput(characters=characters)
        
        