from __future__ import annotations

from typing import Annotated, Any

from base import BaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field

from mongo_client.controller import QAPairHandler

from chat_service.domain.answer_aggregator import (
    AnswerAggregatorInput,
    AnswerAggregatorService,
)
from chat_service.shared.tools import get_context

logger = get_logger(__name__)
class ChatServiceInput(BaseModel):
    question: str
    character_id: str
    character_name: str
    conversation_id: str | None
    
    
class ChatServiceOutput(BaseModel):
    answer: str
    
    
class ChatServiceApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    @property
    def answer_aggregator(self) -> AnswerAggregatorService:
        return AnswerAggregatorService(
            llm=self.request.app.state.llm,
            settings=self.settings.answer_aggregator_settings
        )
    
    async def process(self, input: ChatServiceInput) -> ChatServiceOutput:
        context = await get_context(character_id=input.character_id, question=input.question)
        
        logger.info(f'Total context is: {len(context)}')
        
        if input.conversation_id:
            with self.request.app.state.mongodb_client.get_database() as mongodb:
                try:
                    qa_pair_handler = QAPairHandler(collection=mongodb["qa_pairs"])
                    qa_pairs = qa_pair_handler.get_k_most_recent_qa_pair_by_conversation_id(conversation_id=input.conversation_id, k=3)
                except Exception as e:
                    raise Exception(f"Error accessing MongoDB: {str(e)}")

        answer = await self.answer_aggregator.process(
            inputs=AnswerAggregatorInput(
                question=input.question,
                context=context,
                character_name=input.character_name,
                qa_pairs=qa_pairs
            )
        )
        
        return ChatServiceOutput(answer=answer.answer)