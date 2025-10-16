from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
import uuid

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field

from chat_service.shared.tools import request_livetalking_echo

from .chat_service import ChatServiceApplication, ChatServiceInput

from mongo_client.controller import ConversationHandler, CharacterHandler, QAPairHandler
from mongo_client.model import QAPair

logger = get_logger(__name__)
class QAPairInput(BaseModel):
    conversation_id: str
    
    
class QAPairOutput(BaseModel):
    qa_pairs: list[dict]
    
class CreateQAPairInput(BaseModel):
    question: str
    user_id: str
    conversation_id: str
    sessionid: int

class CreateQAPairOutput(BaseModel):
    answer: str
    
class QAPairService(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    async def process(self, input: QAPairInput) -> QAPairOutput:
        
        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                qa_pair_handler = QAPairHandler(collection=mongodb["qa_pairs"])
                conversations = qa_pair_handler.get_qa_pairs_by_conversation_id(conversation_id=input.conversation_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        return QAPairOutput(qa_pairs=conversations)
    
    async def create_new_qa_pair(self, input: CreateQAPairInput) -> CreateQAPairOutput:
        try:
            chat_service = ChatServiceApplication(
                request=self.request, settings=self.settings
            )
        except Exception as e:
            raise e

        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                # Get conversation
                conversation_handler = ConversationHandler(collection=mongodb["conversations"])
                conversation = conversation_handler.get_conversation_by_id(conversation_id=input.conversation_id)
                # Get character
                character_handler = CharacterHandler(collection=mongodb["characters"])
                character = character_handler.get_character_by_id(character_id=conversation.character_id)

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

                # Request LiveTalking to echo
                request_livetalking_echo(answer, input.sessionid)

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

        return CreateQAPairOutput(answer=answer)