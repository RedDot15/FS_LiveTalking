from __future__ import annotations

import httpx
from collections.abc import Generator
from collections.abc import AsyncGenerator

from .datatypes import CompletionMessage
from .datatypes import Message
from .datatypes import Response
from .datatypes import TokensLLM

from typing import Any
from typing import Dict
from typing import Optional

from base import BaseModel


class LiteLLMChatInput(BaseModel):

    message: Message
    return_type: Any | None = None
    frequency_penalty: Optional[int] = None
    n: Optional[int] = None
    model: str
    presence_penalty: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    tools: Optional[list[dict[str, str | object]]] = None
    count_tokens: bool = False

class LiteLLMChatOutput(BaseModel):
    """
    Output model for LiteLLM service responses.

    Attributes:
        response (Response): The response content from the LLM.
        metadata (dict[str, Any]): Additional metadata about the response.
        count_tokens (bool): Whether tokens were counted in this response.
        tokens (TokensLLM): Token usage information for the request/response.
    """

    response: Response
    metadata: dict[str, Any] = {}
    count_tokens: bool = False
    tokens: TokensLLM = TokensLLM()

class LiteLLMChatService:


    def chat(self, client: Generator[httpx.Client], inputs: LiteLLMChatInput) -> LiteLLMChatOutput:
        """
        chat a synchronous LLM request.

        Args:
            inputs (LiteLLMChatInput): Input parameters for the LLM request.

        Returns:
            LiteLLMChatOutput: The chat output from the LLM.
        """
        return self.__inference_by_llm(
            client=client,
            message=inputs.message,
            return_type=inputs.return_type,
            frequency_penalty=(
                inputs.frequency_penalty if inputs.frequency_penalty else 0
            ),
            n=inputs.n if inputs.n else 1,
            model=inputs.model,
            presence_penalty=inputs.presence_penalty if inputs.presence_penalty else 0,
            count_tokens=inputs.count_tokens,
        )


    async def chat_async(self, client: AsyncGenerator[httpx.AsyncClient], inputs: LiteLLMChatInput) -> LiteLLMChatOutput:
        """
        chat an asynchronous LLM request.

        Args:
            inputs (LiteLLMChatInput): Input parameters for the LLM request.

        Returns:
            LiteLLMOutput: The chated output from the LLM.
        """
        return await self.__inference_by_llm_async(
            client=client,
            message=inputs.message,
            return_type=inputs.return_type,
            frequency_penalty=(
                inputs.frequency_penalty if inputs.frequency_penalty else 0
            ),
            n=inputs.n if inputs.n else 1,
            model=inputs.model,
            presence_penalty=inputs.presence_penalty if inputs.presence_penalty else 0,
            count_tokens=inputs.count_tokens,
        )
        
    def __build_request_payload(
        self,
        message: Message,
        return_type: Any | None,
        frequency_penalty: int,
        n: int,
        model: str,
        presence_penalty: int,
    ) -> Dict[str, Any]:
        """
        Build the request payload for the chat completion API.

        Args:
            message (Message): The message(s) to include in the request.
            return_type (Any | None): Expected response type for structured output.
            frequency_penalty (int): Frequency penalty for token repetition.
            n (int): Number of completions to generate.
            model (str): The model name to use for inference.
            presence_penalty (int): Presence penalty for token usage.

        Returns:
            Dict[str, Any]: The formatted request payload for the API.
        """
        if 'claude' in model.lower():
            payload = {
                'model': model,
                'messages': [self.__parse_to_openai_message(m) for m in message],
                'n': n,
            }
        else:
            payload = {
                'model': model,
                'messages': [self.__parse_to_openai_message(m) for m in message],
                'frequency_penalty': frequency_penalty,
                'presence_penalty': presence_penalty,
                'n': n,
            }

        if return_type:
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': return_type.__name__,
                    'schema': {
                        **return_type.model_json_schema(),
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        return payload
        
    def __inference_by_llm(
        self,
        *,
        client: Generator[httpx.Client],
        message: Message,
        return_type: Any | None,
        frequency_penalty: int,
        n: int,
        model: str,
        presence_penalty: int,
        count_tokens: bool = False,
    ) -> LiteLLMChatOutput:
        """
        Execute synchronous inference using the LLM API.

        Args:
            message (Message): The message(s) to send to the LLM.
            return_type (Any | None): Expected response type for structured output.
            frequency_penalty (int): Frequency penalty for token repetition.
            n (int): Number of completions to generate.
            model (str): The model name to use for inference.
            presence_penalty (int): Presence penalty for token usage.
            count_tokens (bool): Whether to count tokens in the response.

        Returns:
            LiteLLMOutput: The chated response from the LLM.

        Raises:
            httpx.HTTPStatusError: For HTTP-related errors.
            Exception: For other unexpected errors.
        """
        payload = self.__build_request_payload(
            message=message,
            return_type=return_type,
            frequency_penalty=frequency_penalty,
            n=n,
            model=model,
            presence_penalty=presence_penalty,
        )

        try:
            response = client.post('/v1/chat/completions', json=payload)
            response.raise_for_status()
            response_data = response.json()

            return self.__postchating_response(
                response=response_data,
                count_token=count_tokens,
                return_type=return_type,
            )

        except Exception as e:
            raise e

    async def __inference_by_llm_async(
        self,
        *,
        client: AsyncGenerator[httpx.AsyncClient],
        message: Message,
        return_type: Any | None,
        frequency_penalty: int,
        n: int,
        model: str,
        presence_penalty: int,
        count_tokens: bool = False,
    ) -> LiteLLMChatOutput:
        """
        Execute asynchronous inference using the LLM API.

        Args:
            message (Message): The message(s) to send to the LLM.
            return_type (Any | None): Expected response type for structured output.
            frequency_penalty (int): Frequency penalty for token repetition.
            n (int): Number of completions to generate.
            model (str): The model name to use for inference.
            presence_penalty (int): Presence penalty for token usage.
            count_tokens (bool): Whether to count tokens in the response.

        Returns:
            LiteLLMOutput: The chated response from the LLM.

        Raises:
            httpx.HTTPStatusError: For HTTP-related errors.
            Exception: For other unexpected errors.
        """
        payload = self.__build_request_payload(
            message=message,
            return_type=return_type,
            frequency_penalty=frequency_penalty,
            n=n,
            model=model,
            presence_penalty=presence_penalty,
        )

        try:
            response = await client.post('/v1/chat/completions', json=payload)
            response.raise_for_status()
            response_data = response.json()

            return self.__postchating_response(
                response=response_data,
                count_token=count_tokens,
                return_type=return_type,
            )

        except Exception as e:
            raise e
        
    def __parse_to_openai_message(self, message: CompletionMessage) -> Dict[str, str]:
        """
        Parse CompletionMessage to OpenAI API message format.

        Args:
            message (CompletionMessage): CompletionMessage object to convert.

        Returns:
            Dict[str, str]: OpenAI API message format with role and content fields.
        """
        return {
            'role': message.role.value,
            'content': message.content,
        }

    def __postchating_response(
        self,
        response: Dict[str, Any],
        count_token: bool,
        return_type: Any | None,
    ) -> LiteLLMChatOutput:
        """
        Post-chat the response from chat completion API.

        Args:
            response (Dict[str, Any]): The response object received from the LLM API.
            count_token (bool): Flag indicating whether to count tokens used in the response.
            return_type (Any | None): The expected return type for the response. If provided, the response will be validated against this type.

        Returns:
            LiteLLMOutput: The chated output containing the response content, token count, and any tokens used in the completion.

        Raises:
            ValueError: If the response content is empty.
        """
        if not response.get('choices') or not response['choices']:
            raise ValueError('No choices returned in response')

        choice = response['choices'][0]
        content = choice.get('message', {}).get('content')

        if not content:
            raise ValueError('Response returned by client is empty')

        tokens = TokensLLM()
        if count_token and response.get('usage'):
            usage = response['usage']
            tokens.completion_tokens = usage.get('completion_tokens', 0)
            tokens.prompt_tokens = usage.get('prompt_tokens', 0)
            tokens.total_tokens = usage.get('total_tokens', 0)

        return LiteLLMChatOutput(
            response=(
                content
                if not return_type
                else return_type.model_validate_json(
                    content,
                )
            ),
            count_tokens=count_token,
            tokens=tokens,
        )