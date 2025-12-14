from __future__ import annotations

import queue
from queue import Queue
from .models import State

import queue
from queue import Queue
from io import BytesIO
from threading import Thread

from logger import get_logger

logger = get_logger(__name__)

class BaseTTS:
    def __init__(self, parent):
        # Initialize with options and parent object
        self.parent = parent
        
        # Set audio parameters
        self.fps = 50  # 50 fps for 20ms audio frames (optimal for Opus)
        
        self.sample_rate = 16000
        
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms @ 16kHz - optimal for Opus)
        
        self.input_stream = BytesIO()

        # Queue for text messages to be processed
        self.msgqueue = Queue()
        # Current state of the TTS processor
        self.state = State.RUNNING

    def flush_talk(self):
        # Clear the message queue and pause the TTS process
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg:str,eventpoint=None): 
        # Add a text message to the queue for processing
        if len(msg) > 0:
            self.msgqueue.put((msg,eventpoint))

    def render(self,quit_event):
        # Start the TTS processing thread
        process_thread = Thread(target = self.process_tts, args = (quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):     
        # Main loop for the TTS processing thread 
        while not quit_event.is_set():
            try:
                # Get a message from the queue, with a timeout
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                # Continue if the queue is empty
                continue

            # Process the text message into audio
            self.txt_to_audio(msg)
        
        # Log when the TTS processing thread stops
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self, msg):
        # Placeholder method for converting text to audio
        pass