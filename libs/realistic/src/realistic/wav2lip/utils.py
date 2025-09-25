from __future__ import annotations

import os
import torch
import glob
import pickle
from tqdm import tqdm
import cv2
import numpy as np
import time

import queue

from .models import Wav2Lip
from logger import get_logger

logger = get_logger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load(checkpoint_path):
    # Loads checkpoint to CUDA
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path) #,weights_only=True  
    # Loads checkpoint to CPU or MPS if not CUDA.
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location = lambda storage, loc: storage)
	return checkpoint

# Loads the Wav2Lip model from a checkpoint file.
def load_model(path):
    # Instantiates the Wav2Lip model.
	model = Wav2Lip()
	logger.info("Load checkpoint from: {}".format(path))
    # Loads the checkpoint.
	checkpoint = _load(path)
    # Gets the state dictionary.
	s = checkpoint["state_dict"]
    # Creates a new dictionary for the state dict.
	new_s = {}
    # Removes the 'module.' prefix from state dict keys for compatibility with a non-DataParallel model.
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
    # Loads the cleaned state dict into the model.
	model.load_state_dict(new_s)

    # Moves the model to the selected device.
	model = model.to(device)
    # Returns the model in evaluation mode.
	return model.eval()

# Loads avatar assets including full images, face images, and facial coordinates.
def load_avatar(avatar_id):
    # Constructs the avatar directory path.
    avatar_path = f"./wav2lip_data/avatars/{avatar_id}"
    # Constructs the path for full images.
    full_imgs_path = f"{avatar_path}/full_imgs" 
    # Constructs the path for face images.
    face_imgs_path = f"{avatar_path}/face_imgs" 
    # Constructs the path for coordinates file.
    coords_path = f"{avatar_path}/coords.pkl"
    
    # Loads facial coordinates from a pickle file.
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)

    # Finds and sorts full image files.        
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # Reads and loads the full images.
    frame_list_cycle = read_imgs(input_img_list)

    # Commented out: image cache
        #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    
    # Finds and sorts face image files.        
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # Reads and loads the face images.
    face_list_cycle = read_imgs(input_face_list)

    # Returns the loaded assets.
    return frame_list_cycle, face_list_cycle, coord_list_cycle

# Warms up the model with dummy data to ensure it's ready for inference.
@torch.no_grad()
def warm_up(batch_size,model,modelres):
    logger.info('warmup model...')
    # Creates a batch of dummy images.
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    # Creates a batch of dummy mel spectrograms.
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    # Runs a forward pass with the dummy data.
    model(mel_batch, img_batch)

# Reads a list of image paths and returns them as a list of OpenCV image objects.
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')

    # Iterates through image paths with a progress bar.
    for img_path in tqdm(img_list):
        # Reads each image.
        frame = cv2.imread(img_path)
        # Appends the read frame to the list.
        frames.append(frame)

    # Returns the list of frames.
    return frames

# Calculates a mirrored index within a given size, creating a back-and-forth cycle.
def __mirror_index(size, index):
    # Calculates how many full turns have occurred.
    turn = index // size
    # Calculates the remainder within the current turn.
    res = index % size
    if turn % 2 == 0:
        # Return the remainder directly.
        return res
    else: 
        # Return the mirrored index.
        return size - res - 1 

# Inference loop that processes audio features, generates new lip-synced video frames,
# and sends them to a queue.
def inference(quit_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, model):
    
    # model = load_model("./models/wav2lip.pth")
    # input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # face_list_cycle = read_imgs(input_face_list)
    
    # input_latent_list_cycle = torch.load(latents_out_path)

    # Gets the number of face images.
    length = len(face_list_cycle)
    # Initializes the frame index.
    index = 0
    # Initializes a counter for FPS calculation.
    count = 0
    # Initializes a timer for FPS calculation.
    counttime = 0
    logger.info('start inference')

    # Loops until the quit event is set.
    while not quit_event.is_set():
        # Records the start time of the loop.
        starttime = time.perf_counter()
        # Initializes a list for the mel spectrogram batch.
        mel_batch = []
        try:
             # Gets a mel spectrogram batch from the queue.
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            # If the queue is empty, continue to the next iteration.
            continue
            
        # Flag to check if all audio frames are silent.
        is_all_silence=True
        # Initializes a list for audio frames and their metadata.
        audio_frames = []
        # Gets audio frames corresponding to the mel spectrogram batch.
        for _ in range(batch_size*2):
            # Gets audio frame and metadata.
            frame, type, eventpoint = audio_out_queue.get()
            # Appends the audio frame to the list.
            audio_frames.append((frame,type,eventpoint))
            # If any frame is not silent, the whole batch is not silent.
            if type == 0:
                is_all_silence=False

        # If the audio is silent, push placeholder frames and use the full image.
        if is_all_silence:
            for i in range(batch_size):
                # Puts a None frame (signifying silent), the mirrored index, and audio data into the result queue.
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                # Increments the index.
                index = index + 1
        else:
            # Records time for inference duration.
            t=time.perf_counter()
            # Initializes a list for the image batch.
            img_batch = []

            # Populates the image batch from the face list.
            for i in range(batch_size):
                # Gets a mirrored index for the face image.
                idx = __mirror_index(length,index+i)
                # Retrieves the face image.
                face = face_list_cycle[idx]
                # Appends the face image to the batch.
                img_batch.append(face)

            # Converts lists to numpy arrays.
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # Creates a copy of the image batch.
            img_masked = img_batch.copy()
            # Masks the bottom half of the face images.
            img_masked[:, face.shape[0]//2:] = 0

            # Concatenates the masked image and the original image, then normalizes.
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            # Reshapes the mel spectrogram batch.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            # Converts numpy arrays to PyTorch tensors and moves them to the device.
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            # Disables gradient calculation for inference.
            with torch.no_grad():
                # Performs the Wav2Lip inference.
                pred = model(mel_batch, img_batch)

            # Post-processes the prediction: moves to CPU, converts to numpy, transposes, and scales to 0-255.
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            # Updates the total inference time.
            counttime += (time.perf_counter() - t)
            # Updates the frame count
            count += batch_size
            #_totalframe += 1

            # Prints average FPS every 100 frames.
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0

            # Pushes each generated frame to the result queue.
            for i, res_frame in enumerate(pred):
                # Commented out
                    #self.__pushmedia(res_frame,loop,audio_track,video_track)

                # Puts the generated frame, mirrored index, and audio data into the result queue.
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1

    # Logs the stop of the inference process.
    logger.info('lipreal inference processor stop')