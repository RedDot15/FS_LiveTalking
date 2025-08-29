from __future__ import annotations

from base import BaseModel

class PostgresSettings(BaseModel):
    username: str
    password: str
    host: str
    db: str
    