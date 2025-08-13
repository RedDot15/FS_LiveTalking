from __future__ import annotations

from base import BaseModel
from base import BaseService
from chat_service.domain.answer_aggregator import AnswerAggregatorInput
from chat_service.domain.answer_aggregator import AnswerAggregatorService
from chat_service.shared.tools import get_context

from typing import Annotated
from typing import Any

from pydantic import ConfigDict
from pydantic import Field


class ChatServiceInput(BaseModel):
    question: str
    
    
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
        context = await get_context(question=input.question)
        
        answer = await self.answer_aggregator.process(
            inputs=AnswerAggregatorInput(
                question=input.question,
                context=context
            )
        )
        
        return ChatServiceOutput(answer=answer)
        
        