from __future__ import annotations

from ..base import BaseModel
from typing import Optional

class SadTalkerSettings(BaseModel):
    result_dir: str = './results'
    ref_eyeblink: Optional[str] = None
    ref_pose: Optional[str] = None
    checkpoint_dir: str = './checkpoints/'
    config_dir: str = 'shared/tools/config'
    pose_style: int = 0
    batch_size: int = 2
    size: int = 256
    expression_scale: float = 1.0
    input_yaw: Optional[int] = None
    input_pitch: Optional[int] = None
    input_roll: Optional[int] = None
    enhancer: Optional[str] = None
    preprocess: str = 'crop'
    net_recon: str = 'resnet50'
    init_path: Optional[str] = None
    use_last_fc: bool = False
    bfm_folder: str = './checkpoints/BFM_Fitting/'
    bfm_model: str = 'BFM_model_front.mat'
    focal: float = 1015.0
    center: float = 112.0
    camera_d: float = 10.0
    z_near: float = 5.0
    z_far: float = 15.0
    device: str = 'cuda:0'
    face3dvis: bool = False
    still: bool = False
    background_enhancer: Optional[str] = None
    verbose: bool = False