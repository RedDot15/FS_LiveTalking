from __future__ import annotations

from base import BaseModel


class AnswerAggregatorSettings(BaseModel):
    model: str
    context_window: int
