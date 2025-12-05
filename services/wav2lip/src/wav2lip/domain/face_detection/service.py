from __future__ import annotations

from base import BaseModel
from base import BaseService

import torch
import cv2
import numpy as np
from tqdm import tqdm

from .utils import FaceAlignment
from .utils import LandmarksType

class FaceDetectionInput(BaseModel):
    images: list
    batch_size: int
    pads: list
    nosmooth: bool
    
class FaceDetectionOutput(BaseModel):
    faces: list 
    
    
class FaceDetectionService(BaseService):
    
    def process(self, input: FaceDetectionInput) -> FaceDetectionOutput:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        detector = FaceAlignment(
                LandmarksType._2D, 
                flip_input=False, 
                device=device
            )
        
        while 1:
            predictions = []
            try:
                # Ensure all images have the same shape (height, width, channels)
                normalized_images = []
                target_shape = None
                for img in input.images:
                    if img is None:
                        normalized_images.append(None)
                        continue
                    # Convert grayscale to BGR
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # Ensure 3 channels
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    if target_shape is None:
                        target_shape = (img.shape[0], img.shape[1])
                    # Resize to target shape if different
                    if (img.shape[0], img.shape[1]) != target_shape:
                        img = cv2.resize(img, (target_shape[1], target_shape[0]))
                    normalized_images.append(img)

                for i in tqdm(range(0, len(normalized_images), input.batch_size)):
                    batch = [x for x in normalized_images[i:i + input.batch_size] if x is not None]
                    if len(batch) == 0:
                        # extend with None for frames where image was None
                        predictions.extend([None] * min(input.batch_size, len(normalized_images[i:i + input.batch_size])))
                        continue
                    predictions.extend(detector.get_detections_for_batch(np.array(batch)))
            except RuntimeError:
                if input.batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                input.batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(input.batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = input.pads
        for rect, image in zip(predictions, input.images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not input.nosmooth: 
            boxes = self._get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(input.images, boxes)]

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