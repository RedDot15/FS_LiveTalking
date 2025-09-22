from __future__ import annotations

from base import CustomBaseModel
from pydantic import Field


class AnswerAggregatorModel(CustomBaseModel):
    answer: str = Field(
        description="The generated answer to the user's question based on the provided context.",
    )

    able_to_answer: bool = Field(
        description='Indicates whether the system was able to provide a meaningful answer',
    )
