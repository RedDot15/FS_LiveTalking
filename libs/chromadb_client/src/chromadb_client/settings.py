from __future__ import annotations

from chromadb.config import Settings
from base import BaseModel

class ChromaDBSetting(BaseModel):
    host: str
    port: str
    document_collections: str
    allow_reset: bool = True
    anonymized_telemetry: bool = True
