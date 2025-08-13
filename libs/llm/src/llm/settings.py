from __future__ import annotations

from base import BaseModel

class LLMSetting(BaseModel):
    open_ai_key: str
    model_name: str

        