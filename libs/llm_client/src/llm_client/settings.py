from __future__ import annotations

from base import CustomBaseModel

class LLMSetting(CustomBaseModel):
    openai_key: str
    model_name: str

        