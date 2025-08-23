from __future__ import annotations

from base import BaseModel

from text_to_speech import XTTSSettings

class LipRealSettings(BaseModel):
    W: int = 450
    H: int = 450
    fps: int = 20
    idx: int = 0
    batch_size: int = 16
    model: str = 'wav2lip'
    avatar_id: str = ''
    customopt: list = []
    customvideo_config: str = ''
    
    xtts: XTTSSettings = XTTSSettings()
    
    # avatar: tuple[list, list, Any]
    
    # frame_list_cycle: list = []
    # face_list_cycle: list = []
    # coord_list_cycle: Any = []