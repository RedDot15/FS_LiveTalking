from __future__ import annotations

from base import BaseModel, BaseService
from fastapi.encoders import jsonable_encoder
from llm_client import LLMService, LLMServiceInput, MessageRole
from logger import get_logger
from litellm import LiteLLMChatInput, LiteLLMService

from chat_service.shared.models import AnswerAggregatorModel, NoSummaryAnswerAggregatorModel
from chat_service.shared.settings import AnswerAggregatorSettings

from .prompt import ANSWER_AGGREGATOR_SYSTEM_PROMPT, ANSWER_AGGREGATOR_USER_PROMPT, NO_SUMMARY_ANSWER_AGGREGATOR_SYSTEM_PROMPT

logger = get_logger(__name__)



class AnswerAggregatorInput(BaseModel):
    question: str
    context: list[str]
    character_name: str
    qa_pairs: list | None
    language: str = 'VIETNAMESE'
    add_summary: bool
    
class AnswerAggregatorOutput(BaseModel):
    answer: str
    able_to_answer: bool
    conversation_summary: str | None
    
class AnswerAggregatorService(BaseService):
    
    litellm: LiteLLMService
    settings: AnswerAggregatorSettings
    
    async def process(self, inputs: AnswerAggregatorInput) -> AnswerAggregatorOutput:
        
        message = self.build_conversation(
            context="\n\n".join(inputs.context[:self.settings.context_window]),
            raw_question=inputs.question,
            character_name=inputs.character_name,
            language=inputs.language,
            qa_pairs=inputs.qa_pairs,
            add_summary=inputs.add_summary
        )

        logger.info(f'message: {message}')
        
        async with self.litellm.async_client as client:
            if inputs.add_summary:
                response = await self.litellm.chat_async(
                    client=client,
                    inputs=LiteLLMChatInput(
                        message=message,
                        return_type=AnswerAggregatorModel,
                        model=self.litellm.model,
                    ),
                )
            else:
                response = await self.litellm.chat_async(
                    client=client,
                    inputs=LiteLLMChatInput(
                        message=message,
                        return_type=NoSummaryAnswerAggregatorModel,
                        model=self.litellm.model,
                    ),
                )
        
        # TODO: self._create_empty_output()
        if not response:
            return self._create_empty_output()

        if inputs.add_summary:
            answer_aggregator_output = AnswerAggregatorModel(
                **jsonable_encoder(response.response),
            )
        else:
            answer_aggregator_output = NoSummaryAnswerAggregatorModel(
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
            conversation_summary=answer_aggregator_output.conversation_summary if hasattr(answer_aggregator_output, 'conversation_summary') else None
        )
        
    
    def build_conversation(
        self,
        context: str,
        raw_question: str,
        character_name: str,
        qa_pairs: list,
        language: str,
        add_summary: str
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
        if add_summary:
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
                        qa_pairs=qa_pairs
                    ),
                },
            ]
        else:
            return [
                {
                    'role': MessageRole.SYSTEM,
                    'content': NO_SUMMARY_ANSWER_AGGREGATOR_SYSTEM_PROMPT.format(
                        language=language, 
                        character_name=character_name
                    ),
                },
                {
                    'role': MessageRole.USER,
                    'content': ANSWER_AGGREGATOR_USER_PROMPT.format(
                        context=context,
                        raw_question=raw_question,
                        qa_pairs=qa_pairs
                    ),
                },
            ]
