from __future__ import annotations

from base import BaseModel
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


from openai import OpenAI, AsyncOpenAI

class LLMServiceInput(BaseModel):
    message: Message
    model: str
    return_type: type[BaseModel] | None = None
    
class LLMServiceOutput(BaseModel):
    response: Response
    
class LLMService(BaseService):
    settings: LLMSetting
    
    @property
    @contextmanager
    def client(self) -> Generator[OpenAI, None, None]:
        client = OpenAI(api_key=self.settings.open_ai_key)
        try:
            yield client
        except Exception as e:
            raise e
        finally:
            client.close()
            
    @property
    @asynccontextmanager
    async def aclient(self) -> AsyncGenerator[AsyncOpenAI, None]:
        client = AsyncOpenAI(api_key=self.settings.open_ai_key)
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
            return completion

    def __build_request_payload(
        self,
        message: Message,
        model: str,
        return_type: type[BaseModel] | None,
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