from __future__ import annotations

from base import CustomBaseModel
from base import BaseService
from contextlib import contextmanager
from contextlib import asynccontextmanager 
from collections.abc import Generator
from collections.abc import AsyncGenerator

from .datatypes import CompletionMessage
from .datatypes import Message
from .datatypes import Response
from .datatypes import TokensLLM
from .settings import LLMSetting

from typing import Dict
from typing import Any
from typing import cast

from openai import OpenAI, AsyncOpenAI

class LLMServiceInput(CustomBaseModel):
    message: Message
    model: str
    return_type: type[CustomBaseModel] | None = None
    
class LLMServiceOutput(CustomBaseModel):
    response: Response
    
class LLMService(BaseService):
    settings: LLMSetting
    
    @property
    @contextmanager
    def client(self) -> Generator[OpenAI, None, None]:
        client = OpenAI(api_key=self.settings.openai_key)
        try:
            yield client
        except Exception as e:
            raise e
        finally:
            client.close()
            
    @property
    @asynccontextmanager
    async def aclient(self) -> AsyncGenerator[AsyncOpenAI, None]:
        client = AsyncOpenAI(api_key=self.settings.openai_key)
        try:
            yield client
        except Exception as e:
            raise e
        finally:
            await client.close()
    
    def process(self, input: LLMServiceInput) -> LLMServiceOutput:
        payload = self.__build_request_payload(
            message=input.message,
            return_type=input.return_type,
            model=input.model
        )
        
        payload.update({
            "stream": True,
            "stream_options": {"include_usage": True}
        })
        
        with self.client as client:
            completion = client.chat.completions.create(**payload)
            return completion
        
    async def aprocess(self, input: LLMServiceInput) -> LLMServiceOutput:
        payload = self.__build_request_payload(
            message=input.message,
            return_type=input.return_type,
            model=input.model
        )
        
        payload.update({
            "stream": True,
            "stream_options": {"include_usage": True}
        })
        
        async with self.aclient as client:
            completion = await client.chat.completions.create(**payload)
            contents = ""
            
            async for chunk in completion:
                if len(chunk.choices) > 0:
                    msg = chunk.choices[0].delta.content
                    if msg:
                        contents += msg

            response_data = {
                "choices": [{"message": {"content": contents}}],
                "usage": getattr(completion, "usage", None)
            }
            
            return self.__postprocessing_response(
                response=response_data,
                count_token=True,
                return_type=input.return_type
            )

    def __build_request_payload(
        self,
        message: Message,
        model: str,
        return_type: type[CustomBaseModel] | None,
    ) -> Dict[str, Any]:
        
        if 'gpt' in model.lower():
            payload = {
                'model': model,
                'messages': [self.__parse_to_openai_message(m) for m in message],
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
        
    def __postprocessing_response(
        self,
        response: Dict[str, Any],
        count_token: bool,
        return_type: type[CustomBaseModel] | None,
    ) -> LLMServiceOutput:
        """
        Post-process the response from chat completion API.

        Args:
            response (Dict[str, Any]): The response object received from the LLM API.
            count_token (bool): Flag indicating whether to count tokens used in the response.
            return_type (type[BaseModel] | None): The expected return type for the response. If provided, the response will be validated against this type.

        Returns:
            LiteLLMOutput: The processed output containing the response content, token count, and any tokens used in the completion.

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

        return LLMServiceOutput(
            response=(
                content if not return_type else return_type.model_validate_json(
                    content,
                )
            )
        )
