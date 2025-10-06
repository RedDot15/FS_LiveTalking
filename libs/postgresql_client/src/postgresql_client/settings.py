from __future__ import annotations

from base import BaseModel

class PostgresSettings(BaseModel):
    user: str
    password: str
    host: str
    port: str
    db: str