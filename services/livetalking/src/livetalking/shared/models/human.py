from __future__ import annotations

from enum import Enum

class HumanType(str, Enum):
    ECHO = 'echo'
    CHAT = 'chat'