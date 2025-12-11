from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Character:
    _id: str
    name: str
    created_by: str

@dataclass
class Conversation:
    _id: str
    name: str
    participants_hash: str
    character_id: str
    created_at: datetime

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
    created_by: datetime
    created_at: datetime
    status: str
    evaluated_at: str = None
    approved_by: str = None
    rejected_by: str = None
    reject_reason: str = None

