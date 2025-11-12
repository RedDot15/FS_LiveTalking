from __future__ import annotations

from base import BaseModel
from pydantic import Field


class AnswerAggregatorModel(BaseModel):
    answer: str = Field(
        description="The generated answer to the user's question based on the provided context.",
    )

    able_to_answer: bool = Field(
        description='Indicates whether the system was able to provide a meaningful answer',
    )

    conversation_summary: str = Field(
        description="The generated conversation summary to the user's question based on the provided context.",
    )

class NoSummaryAnswerAggregatorModel(BaseModel):
    answer: str = Field(
        description="The generated answer to the user's question based on the provided context.",
    )

    able_to_answer: bool = Field(
        description='Indicates whether the system was able to provide a meaningful answer',
    )