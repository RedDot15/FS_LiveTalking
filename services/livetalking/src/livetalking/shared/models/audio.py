from __future__ import annotations

from enum import Enum

class RecordType(str, Enum):
    START_RECORD = 'start_record'
    END_RECORD = 'end_record'