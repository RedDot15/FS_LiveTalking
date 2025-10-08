from __future__ import annotations

from typing import Annotated, Any

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field

from mongo_client.controller import ConversationHandler

logger = get_logger(__name__)
class ConversationInput(BaseModel):
    user_id: str
    character_id: str
    
    
class ConversationOutput(BaseModel):
    conversations: list[dict]
    
    
class ConversationService(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    async def process(self, input: ConversationInput) -> ConversationOutput:
        participants_hash = f"{input.user_id}_{input.character_id}"
        
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                conversation_handler = ConversationHandler(collection=mongodb["conversations"])
                conversations = conversation_handler.get_conversation_by_participants_hash(participants_hash=participants_hash)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        return ConversationOutput(conversations=conversations)