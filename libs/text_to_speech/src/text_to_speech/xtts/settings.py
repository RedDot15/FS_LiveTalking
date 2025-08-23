from __future__ import annotations

from base import BaseModel
from typing import Any

from queue import Queue

class XTTSSettings(BaseModel):
    REF_FILE: str = 'thang.wav'
    TTS_SERVER: str = 'http://127.0.0.1:8002'