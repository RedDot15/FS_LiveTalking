from __future__ import annotations

from base import BaseModel
from base import BaseService
from logger import get_logger
from fastapi.encoders import jsonable_encoder

from .prompt import ANSWER_AGGREGATOR_SYSTEM_PROMPT
from .prompt import ANSWER_AGGREGATOR_USER_PROMPT
from chat_service.shared.settings import AnswerAggregatorSettings
from chat_service.shared.models import AnswerAggregatorModel

from llm_client import LLMService
from llm_client import LLMServiceInput
from llm_client import MessageRole


logger = get_logger(__name__)

class AnswerAggregatorInput(BaseModel):
    question: str
    context: list[str]
    character_name: str
    language: str = 'VIETNAMESE'
    
class AnswerAggregatorOutput(BaseModel):

    answer: str
    able_to_answer: bool
    
class AnswerAggregatorService(BaseService):
    
    llm: LLMService
    settings: AnswerAggregatorSettings
    
    async def process(self, inputs: AnswerAggregatorInput) -> AnswerAggregatorOutput:
        
        
        message = self.build_conversation(
            context=inputs.context[: self.settings.context_window],
            raw_question=inputs.question,
            character_name=inputs.character_name,
            language=inputs.language,
        )
        
        response = await self.llm.aprocess(
            LLMServiceInput(
                message=message,
                return_type=AnswerAggregatorModel,
                model=self.settings.model,
                
            ),
        )
        
        if not response:
            return self._create_empty_output()

        answer_aggregator_output = AnswerAggregatorModel(
            **jsonable_encoder(response.response),
        )
        
        logger.info(
            'Answer aggregation processing completed successfully',
            extra={
                'response': jsonable_encoder(answer_aggregator_output),
            },
        )
        
        return AnswerAggregatorOutput(
            answer=answer_aggregator_output.answer,
            able_to_answer=answer_aggregator_output.able_to_answer,
        )
        
    
    def build_conversation(
        self,
        context: str,
        raw_question: str,
        character_name: str,
        language: str
    ) -> list[dict]:
        """
        Build conversation messages for LLM processing.

        This method constructs the conversation structure required by the LLM,
        including system and user messages with appropriate prompts and context.
        It formats both the original and rephrased questions along with the context.

        Args:
            context (str): The combined context string containing relevant information
            raw_question (str): The original user's question to be answered
            rephrase_query (str): The rephrased version of the user's question

        Returns:
            list[dict]: List of message dictionaries with role and content for LLM processing
        """
        return [
            {
                'role': MessageRole.SYSTEM,
                'content': ANSWER_AGGREGATOR_SYSTEM_PROMPT.format(
                    language=language, 
                    character_name=character_name
                ),
            },
            {
                'role': MessageRole.USER,
                'content': ANSWER_AGGREGATOR_USER_PROMPT.format(
                    context=context,
                    raw_question=raw_question,
                ),
            },
        ]
