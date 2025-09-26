from __future__ import annotations

from base import CustomBaseModel

class MinioSettings(CustomBaseModel):
    host: str
    http_port: int
    access_key: str
    secret_key: str
    secure: bool = False
