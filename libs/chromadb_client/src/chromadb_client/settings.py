from __future__ import annotations

from base import BaseModel

class ChromaDBSetting(BaseModel):
    host: str
    port: int
    document_collections: str
    allow_reset: bool = True
    anonymized_telemetry: bool = True
    model_name: str = 'bkai-foundation-models/vietnamese-bi-encoder'