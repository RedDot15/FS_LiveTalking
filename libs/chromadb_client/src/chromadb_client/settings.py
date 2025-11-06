from __future__ import annotations

from base import BaseModel

class ChromaDBSetting(BaseModel):
    host: str
    port: int
    allow_reset: bool = True
    anonymized_telemetry: bool = True