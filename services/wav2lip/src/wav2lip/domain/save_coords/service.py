from __future__ import annotations

from base import BaseModel
from base import BaseService

import cv2
import pickle

class SaveCoordsInput(BaseModel):
    
    img_size: int
    coords_path: str
    face_det_results: list
    face_imgs_path: str
    
    
class SaveCoordsService(BaseService):
    
    def process(self, input: SaveCoordsInput) -> None:
        
        idx = 0
        coord_list = []
        for frame,coords in input.face_det_results:        
            #x1, y1, x2, y2 = bbox
            resized_crop_frame = cv2.resize(frame,(input.img_size, input.img_size)) #,interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(f"{input.face_imgs_path}/{idx:08d}.png", resized_crop_frame)
            coord_list.append(coords)
            idx = idx + 1
        
        with open(input.coords_path, 'wb') as f:
            pickle.dump(coord_list, f)

        