from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Character:
    _id: str
    name: str
    knowledge_file_info: dict
    image_file_info: dict
    audio_file_info: dict

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
    character_name: str
    created_by: datetime
    created_at: datetime
    knowledge_file_info: dict
    image_file_info: dict
    audio_file_info: dict
    status: str
    approved_by: str
    approved_at: str
    rejected_by: str
    reject_reason: str

@dataclass
class Request:
    _id: str
    character_name: str
    created_by: datetime
    created_at: datetime
    knowledge_file_info: dict
    image_file_info: dict
    audio_file_info: dict
    status: str
    approved_by: str
    approved_at: str
    rejected_by: str
    reject_reason: str

class Rating:
    _id: str
    character_id: str
    commented_by: dict
    rating: float
    comment: str
    created_at: datetime
    updated_at: datetime