from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Character:
    _id: str
    name: str

@dataclass
class Conversation:
    name: str
    participants_hash: str
    character_id: str
    created_at: datetime

@dataclass
class QAPair:
    question: str
    answer: str
    response_time: int

    conversation_id: str = field(default=None)
    created_at: datetime = field(default=None)
    updated_at: datetime = field(default=None)

