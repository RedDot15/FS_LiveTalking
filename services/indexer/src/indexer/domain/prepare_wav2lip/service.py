from __future__ import annotations

from base import BaseModel
from base import BaseService

import os
import io
import imageio.v3 as iio
import cv2

import glob
import pickle
from fastapi import UploadFile
import numpy as np
from tqdm import tqdm

from .utils import FaceAlignment
from .utils import LandmarksType
import torch

class PrepareWav2lipInput(BaseModel):
    
    avatar_id: str
    video_file: UploadFile
    img_size: int
    nosmooth: bool
    pads: list
    face_det_batch_size: int
    
class PrepareWav2lipOutput(BaseModel):
    pass 

class PrepareWav2lipService(BaseService):
    
    async def process(self, input: PrepareWav2lipInput) -> PrepareWav2lipOutput:
        avatar_path = f"./data/avatars/{input.avatar_id}"
        full_imgs_path = f"{avatar_path}/full_imgs" 
        face_imgs_path = f"{avatar_path}/face_imgs" 
        coords_path = f"{avatar_path}/coords.pkl"
        
        self._osmakedirs([avatar_path,full_imgs_path,face_imgs_path])
        
        await self._video2imgs(file=input.video_file, save_path=full_imgs_path, ext='.png')
        input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        frames = self._read_imgs(img_list=input_img_list)
        face_det_results = self._face_detect(images=frames, batch_size=input.face_det_batch_size, pads=input.pads, nosmooth=input.nosmooth) 
        coord_list = []
        idx = 0
        for frame,coords in face_det_results:        
            #x1, y1, x2, y2 = bbox
            resized_crop_frame = cv2.resize(frame,(input.img_size, input.img_size)) #,interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
            coord_list.append(coords)
            idx = idx + 1
        
        with open(coords_path, 'wb') as f:
            pickle.dump(coord_list, f)
            
    async def _video2imgs(self, file: UploadFile, save_path: str, ext: str, cut_frame: int = 10_000_000):
        video_bytes = await file.read()
        buffer = io.BytesIO(video_bytes)

        count = 0
        for idx, frame in enumerate(iio.imiter(buffer, plugin="pyav")):
            if count > cut_frame:
                break

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.putText(frame_bgr, "LiveTalking", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)

            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame_bgr)
            count += 1

        return {"frames_saved": count}
    
    def _read_imgs(self, img_list):
        frames = []
        print('reading images...')
        for img_path in tqdm(img_list):
            frame = cv2.imread(img_path)
            frames.append(frame)
        return frames
    
    def _face_detect(self, images: list, batch_size: int, pads: list, nosmooth: bool):
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        detector = FaceAlignment(
                LandmarksType._2D, 
                flip_input=False, 
                device=device
            )
        
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth: 
            boxes = self._get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results 
    
    def _get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
                
    def _osmakedirs(self, path_list):
        for path in path_list:
            os.makedirs(path) if not os.path.exists(path) else None