from __future__ import annotations

from base import BaseModel

class LLMSetting(BaseModel):
    openai_key: str
    model_name: str

        