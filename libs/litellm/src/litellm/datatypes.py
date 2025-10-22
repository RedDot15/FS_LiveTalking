from __future__ import annotations

from enum import Enum

from base import BaseModel
from typing_extensions import Required
from typing_extensions import TypedDict


class TypeMessage(str, Enum):
    TEXT = 'text'
    IMAGE_URL = 'image_url'


class BaseImageUrl(TypedDict, total=False):
    url: Required[str]


class BaseLLMMessage(BaseModel):
    """Base Message for LLM, all message used by LLM should inherit this model"""

    type: TypeMessage = TypeMessage.TEXT
    image_url: BaseImageUrl | None = None
    content: str = ''


class MessageRole(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class CompletionMessage(BaseLLMMessage):
    role: MessageRole


class TokensLLM(BaseModel):
    """Tokens used by LLM"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


Message = list[CompletionMessage]
Response = str | BaseModel
