from __future__ import annotations

from typing import Annotated, Any

from base import CustomBaseModel, BaseService
from logger import get_logger
from pydantic import ConfigDict, Field

from chat_service.domain.answer_aggregator import (
    AnswerAggregatorInput,
    AnswerAggregatorService,
)
from chat_service.shared.tools import get_context

logger = get_logger(__name__)
class ChatServiceInput(CustomBaseModel):
    question: str
    character_name: str
    
    
class ChatServiceOutput(CustomBaseModel):
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
        context = await get_context(question=input.question)
        
        logger.info(f'Total context is: {len(context)}')
        
        answer = await self.answer_aggregator.process(
            inputs=AnswerAggregatorInput(
                question=input.question,
                context=context,
                character_name=input.character_name
            )
        )
        
        return ChatServiceOutput(answer=answer.answer)