from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Character:
    _id: str
    name: str

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
    character_id: str = None
    character_name: str = None
    knowledge_url: str = None
    avatar_url: str = None
    audio_url: str = None
    created_by: datetime = None
    created_at: datetime = None
    status: str = None
    evaluated_at: str = None
    approved_by: str = None
    rejected_by: str = None
    reject_reason: str = None

