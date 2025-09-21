from __future__ import annotations

from shared.base import BaseModel

class MinioSettings(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
