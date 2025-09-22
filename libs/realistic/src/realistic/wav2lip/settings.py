from __future__ import annotations

from base import CustomBaseModel

from text_to_speech import XTTSSettings

class LipRealSettings(CustomBaseModel):
    W: int
    H: int
    fps: int
    idx: int
    batch_size: int
    model: str
    customopt: list
    customvideo_config: str
    
    xtts: XTTSSettings