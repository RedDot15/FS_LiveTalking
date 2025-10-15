from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
import uuid

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field
from fastapi import Request

from chat_service.shared.utils import get_settings
from chat_service.shared.tools import request_livetalking_echo

from .chat_service import ChatServiceApplication, ChatServiceInput

from mongo_client.controller import ConversationHandler, CharacterHandler, QAPairHandler
from mongo_client.model import Conversation, QAPair


logger = get_logger(__name__)

class ConversationInput(BaseModel):
    user_id: str
    character_id: str
    
class ConversationOutput(BaseModel):
    conversations: list[dict]

class CreateConversationInput(BaseModel):
    question: str
    user_id: str
    character_id: str
    sessionid: int

class CreateConversationOutput(BaseModel):
    answer: str
    
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
    
    async def create_new_conversation(self, input: CreateConversationInput) -> CreateConversationOutput:
        try:
            chat_service = ChatServiceApplication(
                request=self.request, settings=self.settings
            )
        except Exception as e:
            raise e
        
        # "Generate" participants_hash
        participants_hash = f"{input.user_id}_{input.character_id}"

        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                # Get character
                character_handler = CharacterHandler(collection=mongodb["characters"])
                character = character_handler.get_character_by_id(character_id=input.character_id)

                # Record start time
                start_time = datetime.now()

                # Get response from llm
                chat_service_output = chat_service.process(ChatServiceInput(
                    question=input.question, 
                    character_id=input.character_id, 
                    character_name=character.name,
                    conversation_id=None))
                answer = chat_service_output.answer

                # Record end time
                end_time = datetime.now()
                # Calculate response duration in seconds (as a float or string)
                response_duration = (end_time - start_time).microseconds() 

                # TODO: Call to LiveTalking
                request_livetalking_echo(answer, input.sessionid)

                # Insert into DB new conversation 
                conversation_handler = ConversationHandler(collection=mongodb["conversations"])
                conversation = conversation_handler.create_conversation(conversation=Conversation(
                    _id=uuid.uuid4(), 
                    name=answer, 
                    participants_hash=participants_hash, 
                    character_id=input.character_id, 
                    created_at=datetime.now()))

                # Insert into DB new qa_pair
                qa_pair_handler = QAPairHandler(collection=mongodb["qa_pairs"])
                qa_pair_handler.create_qa_pair(QAPair(
                    _id=uuid.uuid4(), 
                    question=input.question, 
                    answer=answer, 
                    response_time=str(response_duration), 
                    created_at=datetime.now(), 
                    updated_at=datetime.now(), 
                    conversation_id=conversation._id))
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        return CreateConversationOutput(answer=answer)