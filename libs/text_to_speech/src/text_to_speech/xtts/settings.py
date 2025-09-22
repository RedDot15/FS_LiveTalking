from __future__ import annotations

from base import CustomBaseModel
from typing import Any

from queue import Queue

class XTTSSettings(CustomBaseModel):
    ref_file: str
    tts_server: str