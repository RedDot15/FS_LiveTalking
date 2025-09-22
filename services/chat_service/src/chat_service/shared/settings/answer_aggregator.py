from __future__ import annotations

from base import CustomBaseModel


class AnswerAggregatorSettings(CustomBaseModel):
    model: str
    context_window: int
