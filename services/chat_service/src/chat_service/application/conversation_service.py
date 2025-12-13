from sqlalchemy.sql.elements import Null
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
import uuid

from base import BaseModel, BaseService
from logger import get_logger
from mongo_client.model.entity import Character
from pydantic import ConfigDict, Field

from chat_service.shared.tools import request_livetalking_echo

from .chat_service import ChatServiceApplication, ChatServiceInput, ChatServiceOutput

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
    user_id: str = "default"
    character_id: str
    conversation_id: str = ""

class CreateConversationOutput(BaseModel):
    answer: str
    
class DeleteConversationInput(BaseModel):
    conversation_id: str
    user_id: str = "default"

class DeleteConversationOutput(BaseModel):
    conversation_id: str

class ChatInConversationInput(BaseModel):
    conversation_id: str
    question: str
    user_id: str = "default"

class ChatInConversationOutput(BaseModel):
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
    
    async def _process_chat_interaction(
        self, 
        character: Character, 
        conversation_id: str, 
        question: str, 
        add_summary: bool
    ) -> str:
        try:
            chat_service = ChatServiceApplication(
                request=self.request, settings=self.settings
            )
        except Exception as e:
            raise e

        # Record start time
        start_time = datetime.now()

        # Get response from llm
        chat_service_output: ChatServiceOutput = await chat_service.process(ChatServiceInput(
            question=question, 
            character_id=character['_id'], 
            character_name=character['name'],
            conversation_id=conversation_id,
            add_summary=add_summary))
        answer = chat_service_output.answer
        conversation_summary = chat_service_output.conversation_summary
        
        logger.info(f"answer: {answer}")
        logger.info(f"summary: {conversation_summary}")

        # Record end time
        end_time = datetime.now()
        # Calculate response duration in seconds (as a float or string)
        response_duration = (end_time - start_time).microseconds

        # Request LiveTalking to echo
        await request_livetalking_echo(answer, character['_id'], self.request)

        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                # Insert into DB new qa_pair
                qa_pair_handler = QAPairHandler(collection=mongodb["qa_pairs"])
                qa_pair_handler.create_qa_pair(QAPair(
                    _id=str(uuid.uuid4()), 
                    question=question, 
                    answer=answer, 
                    response_time=str(response_duration), 
                    created_at=datetime.now(), 
                    updated_at=datetime.now(), 
                    conversation_id=conversation_id))
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")
        
        return answer

    async def create_new_conversation(self, input: CreateConversationInput) -> CreateConversationOutput:
        # "Generate" participants_hash
        participants_hash = f"{input.user_id}_{input.character_id}"

        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                # Get character
                character_handler = CharacterHandler(collection=mongodb["characters"])
                character: Character = character_handler.get_character_by_id(character_id=input.character_id)

                conversation_handler = ConversationHandler(collection=mongodb["conversations"])

                if input.conversation_id != "":
                    add_summary = False
                    # Insert into DB new conversation 
                    conversation: Conversation = conversation_handler.create_conversation(conversation=Conversation(
                        _id=str(uuid.uuid4()), 
                        name=input.question[:50],
                        participants_hash=participants_hash, 
                        character_id=input.character_id, 
                        created_at=datetime.now()))
                    conversation_id = conversation.inserted_id
                else:
                    add_summary = False
                    conversation: Conversation = conversation_handler.get_conversation_by_id(conversation_id=input.conversation_id)

                    # Validate conversation owner
                    if conversation['participants_hash'].split('_')[0] != input.user_id:
                        raise Exception(f"Unauthorize user: {input.user_id}")
                    
                    character_id = conversation['character_id']

                    # Get character
                    character_handler = CharacterHandler(collection=mongodb["characters"])
                    character: Character = character_handler.get_character_by_id(character_id=character_id)

            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        answer = await self._process_chat_interaction(
            character=character,
            conversation_id=conversation_id,
            question=input.question,
            add_summary=add_summary
        )

        return CreateConversationOutput(answer=answer)
    
    async def delete_conversation(self, input: DeleteConversationInput) -> DeleteConversationOutput:

        with self.request.app.state.mongodb_client.get_database() as mongodb:
            try:
                # Get character
                conversation_handler = ConversationHandler(collection=mongodb["conversations"])
                conversation: Conversation = conversation_handler.get_conversation_by_id(conversation_id=input.conversation_id)

                # Validate conversation owner
                if conversation['participants_hash'].split('_')[0] != input.user_id:
                    raise Exception(f"Unauthorize user: {input.user_id}")

                conversation_handler.delete_conversation_by_id(conversation_id=input.conversation_id)
            except Exception as e:
                raise Exception(f"Error accessing MongoDB: {str(e)}")

        return DeleteConversationOutput(answer=input.conversation_id)

