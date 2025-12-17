from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Character:
    _id: str
    name: str
    created_by: str
    is_deleted: bool = False

@dataclass
class Conversation:
    _id: str
    name: str
    participants_hash: str
    character_id: str
    created_at: datetime
    is_deleted: bool = False

@dataclass
class QAPair:
    _id: str
    question: str
    answer: str
    response_time: int
    created_at: datetime = None
    updated_at: datetime = None
    conversation_id: str = None

@dataclass
class Request:
    _id: str
    character_id: str
    character_name: str
    knowledge_url: str
    avatar_url: str
    audio_url: str
    created_by: str
    created_at: datetime
    status: str
    evaluated_at: datetime = None
    approved_by: str = None
    rejected_by: str = None
    reject_reason: str = None
    is_deleted: bool = False

class Rating:
    _id: str
    character_id: str
    commented_by: dict
    rating: float
    comment: str
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False