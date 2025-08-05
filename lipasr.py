import time
import torch
import numpy as np

import queue
from queue import Queue
#import multiprocessing as mp

from baseasr import BaseASR
from wav2lip import audio

# A class for Automatic Speech Recognition (ASR) specifically tailored for
# lip-sync tasks, inheriting from BaseASR. It processes audio frames
# to extract mel spectrogram features.
class LipASR(BaseASR):

    # Processes a step of audio data to extract mel spectrogram features for lip-sync.
    def run_step(self):
        ############################################## extract audio feature ##############################################
        # get a frame of audio
        # The loop collects a batch of audio frames.
        for _ in range(self.batch_size*2):
            # Retrieves an audio frame, its type, and an event point from a queue (not shown here).
            frame, type, eventpoint = self.get_audio_frame()
            # Appends the audio frame to a list of frames.
            self.frames.append(frame)
            # put to output
            # Puts the processed frame, type, and event point into an output queue.
            self.output_queue.put((frame, type, eventpoint))

        
        # Checks if there are enough frames to form a context window for feature extraction.
        # If not: context not enough, do not run network.
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        # Concatenates the collected frames into a single numpy array.
        inputs = np.concatenate(self.frames) # [N * chunk]
        # Extracts the mel spectrogram from the concatenated audio input.
        mel = audio.melspectrogram(inputs)
        #print(mel.shape[0],mel.shape,len(mel[0]),len(self.frames))
    
        # cut off stride
        # Calculates the starting index for the mel spectrogram, considering the left stride.
        left = max(0, self.stride_left_size*80/50)
        # Calculates the ending index for the mel spectrogram, considering the right stride.
        right = min(len(mel[0]), len(mel[0]) - self.stride_right_size*80/50)
        # Sets the multiplier to determine the step size for mel spectrogram chunks.
        mel_idx_multiplier = 80.*2/self.fps 
        # Defines the size of each mel spectrogram chunk.
        mel_step_size = 16

        # Initializes a counter.
        i = 0
        # Creates a list to store the mel spectrogram chunks.
        mel_chunks = []
        # Loops through the frames to create mel spectrogram chunks.
        while i < (len(self.frames)-self.stride_left_size-self.stride_right_size)/2:
            # Calculates the starting index for the current chunk.
            start_idx = int(left + i * mel_idx_multiplier)
            # If the chunk goes past the end, it takes the last `mel_step_size` frames.
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            # Otherwise, it extracts the chunk normally.
            else:
                mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        
        # Puts the list of mel spectrogram chunks into a feature queue.
        self.feat_queue.put(mel_chunks)
        
        # discard the old part to save memory
        # Discards the old frames to free up memory, keeping only the necessary context for the next step.
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
