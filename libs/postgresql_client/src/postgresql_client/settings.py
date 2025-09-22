from __future__ import annotations

from base import CustomBaseModel


class PostgresSettings(CustomBaseModel):
    user: str
    password: str
    host: str
    port: str
    db: str
