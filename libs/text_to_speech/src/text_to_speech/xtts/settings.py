from __future__ import annotations

from base import BaseModel
from typing import Any

from queue import Queue

class XTTSSettings(BaseModel):
    ref_file: str
    tts_server: str