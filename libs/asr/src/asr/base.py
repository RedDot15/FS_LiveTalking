import time
import numpy as np

import queue
from queue import Queue
import torch.multiprocessing as mp

class BaseASR:
    def __init__(self, parent):
        self.parent = parent
        
        # 1000/20 frame per second
        self.fps = 20
        # 16000 sample per second
        self.sample_rate = 16000 
        # Calculates the number of audio samples per frame 320 samples per frame (20s * 16000 / 1000)
        self.chunk = self.sample_rate // self.fps 

        # Initializes a standard (thread-safe) queue for incoming audio frames.
        self.queue = Queue()
        # Initializes a multiprocessing queue for output, enabling inter-process communication.
        self.output_queue = mp.Queue()

        self.batch_size = 16

        # Initializes an empty list to store audio frames.
        self.frames = []
        self.stride_left_size = 10
        self.stride_right_size = 10

        # (This line is commented out, suggesting it's an unused or experimental variable).
            #self.context_size = 10

        # Initializes a multiprocessing queue for features with a maximum size of 2.
        self.feat_queue = mp.Queue(2)

    # Defines a method to clear the internal audio queue.
    def flush_talk(self):
        self.queue.queue.clear()

    # Defines a method to add an audio chunk to the queue.
    def put_audio_frame(self,audio_chunk,eventpoint=None): # expected format of the audio_chunk: 16khz 20ms pcm
        self.queue.put((audio_chunk,eventpoint))

    # Defines a method to retrieve an audio frame from the queue.
    # return frame:
        # audio pcm; 
        # type: 0-normal_speak, 1-silence; 
        # eventpoint: custom event sync with audio
    def get_audio_frame(self):        
        try:
            frame, eventpoint = self.queue.get(block=True,timeout=0.01)
            # If successful, sets the type to 0 (normal speak).
            type = 0
        except queue.Empty:
            # Checks if a parent object exists and its current state is greater than 1 (likely indicating a custom audio playback state).
            if self.parent and self.parent.curr_state > 1: 
                # Retrieves an audio stream from the parent based on its current state.
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                # Sets the type to the parent's current state.
                type = self.parent.curr_state
            # If no parent or the state is not for custom audio.
            else:
                # Creates an array of zeros representing a silence frame.
                frame = np.zeros(self.chunk, dtype = np.float32)
                # Sets the type to 1 (silence).
                type = 1
            # Sets eventpoint to None as there's no specific event for silence or custom audio.
            eventpoint = None

        return frame, type, eventpoint 

    # Defines a method to retrieve processed audio output.
    # return frame:
        # audio pcm; 
        # type: 0-normal speak, 1-silence; 
        # eventpoint:custom event sync with audio
    def get_audio_out(self): 
        return self.output_queue.get()
    
    # Defines a warm-up method to pre-fill frames and queues.
    def warm_up(self):
        # Loops a number of times equal to the sum of left and right stride sizes.
        for _ in range(self.stride_left_size + self.stride_right_size):
            # Gets an audio frame, its type, and eventpoint.
            audio_frame, type, eventpoint = self.get_audio_frame()
            # Appends the audio frame to the internal frames list.
            self.frames.append(audio_frame)
            # Puts the audio frame, type, and eventpoint into the output queue.
            self.output_queue.put((audio_frame, type, eventpoint))
        # Loops a number of times equal to the left stride size.
        for _ in range(self.stride_left_size):
            # Removes items from the output queue (likely to ensure proper initial state for processing).
            self.output_queue.get()

    # Defines a placeholder method for a single step of the ASR process.
    def run_step(self):
        pass # Does nothing, as it's meant to be overridden by subclasses.

    # Defines a method to retrieve the next feature from the feature queue.
    def get_next_feat(self, block, timeout):        
        # Gets an item from the feature queue, with options for blocking and timeout.
        return self.feat_queue.get(block,timeout)