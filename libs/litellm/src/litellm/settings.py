from __future__ import annotations

from base import BaseModel
from pydantic import HttpUrl


class LiteLLMSetting(BaseModel):
    url: HttpUrl
    model: str
    embedding_model: str
    frequency_penalty: int
    n: int
    presence_penalty: int
    temperature: float
    top_p: float
    max_completion_tokens: int
    encoding_format: str
    dimensions: int
    max_length: int
