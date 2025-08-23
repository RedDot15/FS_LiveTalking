from __future__ import annotations

import cv2
from tqdm import tqdm


def read_imgs(img_list):
    frames = []
    # Iterates through image paths with a progress bar.
    for img_path in tqdm(img_list):
        # Reads an image using OpenCV.
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames