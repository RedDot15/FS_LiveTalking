from __future__ import annotations

from base import BaseModel
from base import BaseService
from logger import get_logger
from fastapi.encoders import jsonable_encoder

from .prompt import ANSWER_AGGREGATOR_SYSTEM_PROMPT
from .prompt import ANSWER_AGGREGATOR_USER_PROMPT
from chat_service.shared.settings import AnswerAggregatorSettings
from chat_service.shared.models import AnswerAggregatorModel

from llm import LLMService
from llm import LLMServiceInput
from llm import MessageRole


logger = get_logger(__name__)

class AnswerAggregatorInput(BaseModel):
    question: str
    context: list[str]
    language: str = 'VIETNAMESE'
    
class AnswerAggregatorOutput(BaseModel):

    answer: str
    able_to_answer: bool
    
class AnswerAggregatorService(BaseService):
    
    llm: LLMService
    settings: AnswerAggregatorSettings
    
    async def process(self, inputs: AnswerAggregatorInput) -> AnswerAggregatorOutput:
        
        joined_contexts = self._join_contexts_with_questions(
            inputs.context,
            inputs.question,
        )
        
        message = self.build_conversation(
            context=joined_contexts[: self.settings.context_window],
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
                'content': ANSWER_AGGREGATOR_SYSTEM_PROMPT.format(language=language),
            },
            {
                'role': MessageRole.USER,
                'content': ANSWER_AGGREGATOR_USER_PROMPT.format(
                    context=context,
                    raw_question=raw_question,
                ),
            },
        ]
        
    def _join_contexts_with_questions(
        self,
        contexts: list[str],
        questions: list[str],
    ) -> str:
        """
        Join contexts with their corresponding questions.

        This method combines context strings with their related sub-questions,
        formatting them with markdown headers for better structure and readability
        when passed to the LLM.

        Args:
            contexts (list[str]): List of context strings to be combined
            questions (list[str]): List of corresponding sub-questions

        Returns:
            str: Formatted string with questions as headers followed by their contexts
        """
        joined_contexts = ''
        for context, question in zip(contexts, questions):
            if (
                not context
                == 'Không có thông tin liên quan đến câu hỏi trong các chunk đã cung cấp.'
            ):
                joined_contexts += ''
            joined_contexts += f'### {question}\n\n{context}\n\n'
        return joined_contexts