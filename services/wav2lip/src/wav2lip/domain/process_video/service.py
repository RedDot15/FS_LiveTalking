from __future__ import annotations

from base import BaseModel
from base import BaseService

import os
import cv2
import glob
from tqdm import tqdm

from logger import get_logger

logger = get_logger(__name__)

class ProcessVideoInput(BaseModel):
    
    video_path: str
    save_path: str
    ext: str
    cut_frame: int
    
class ProcessVideoOutput(BaseModel):
    frames: list
    
class ProcessVideoService(BaseService):
    
    def process(self, input: ProcessVideoInput) -> ProcessVideoOutput:
        cap = cv2.VideoCapture(input.video_path)
        count = 0
        while True:
            if count > input.cut_frame:
                break
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "Reunion", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
                cv2.imwrite(f"{input.save_path}/{count:08d}.png", frame)
                count += 1
            else:
                break


        input_img_list = sorted(glob.glob(os.path.join(input.save_path, '*.[jpJP][pnPN]*[gG]')))
        frames = self._read_imgs(img_list=input_img_list)
        
        return ProcessVideoOutput(
            frames=frames
        )
        
    def _read_imgs(self, img_list: list) -> list:
        frames = []
        logger.info('READING IMAGES...')
        for img_path in tqdm(img_list):
            frame = cv2.imread(img_path)
            frames.append(frame)
        return frames