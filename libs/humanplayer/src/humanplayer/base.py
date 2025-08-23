from __future__ import annotations

import time
import fractions
from typing import Tuple
from typing import Union
from av.frame import Frame
from av.packet import Packet

from base import BaseModel
from logger import get_logger


import asyncio
from aiortc import MediaStreamTrack

logger = get_logger(__name__)

class PlayerStreamTrack(MediaStreamTrack):
    
    """
    The constructor initializes the track, setting its kind ('audio' or 'video') and linking it to the parent HumanPlayer. 
    It creates an asynchronous queue to hold media frames and sets up counters for performance tracking.
    """
    def __init__(self, player, kind):
        # Initializes the parent MediaStreamTrack class.
        super().__init__()
        # Stores the type of track ('audio' or 'video').  
        self.kind = kind 
        # Stores a reference to the parent player object.
        self._player = player
        # Creates an asynchronous queue to hold media frames. 
        self._queue = asyncio.Queue() 
        # A list to record timestamps of recent packets (currently commented out/unused).
        self.timelist = [] 
        # Counter for the number of frames processed.
        self.current_frame_count = 0

        # If the track is for video:
        if self.kind == 'video':
            # Initializes a counter for frames sent for FPS calculation.
            self.framecount = 0
            # Records the high-resolution performance counter time.
            self.lasttime = time.perf_counter()
            # Initializes a variable to sum up the time between frames.
            self.totaltime = 0
            
        self.VIDEO_PTIME = 0.040 
        self.VIDEO_CLOCK_RATE = 90000
        self.VIDEO_TIME_BASE = fractions.Fraction(1, self.VIDEO_CLOCK_RATE) 
        self.AUDIO_PTIME = 0.020 
        self.SAMPLE_RATE = 16000
        self.AUDIO_TIME_BASE = fractions.Fraction(1, self.SAMPLE_RATE) 
        
    
    # Type hint for the start time of the stream.
    _start: float 
    # Type hint for the current timestamp.
    _timestamp: int

    """
    This is an asynchronous function that calculates and returns the next timestamp for a media frame. 
    It ensures a consistent frame rate by calculating the time to wait before the next frame is due and pausing the execution if necessary.
    """
    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        # Checks if the stream is in a 'live' state.
        # Raises an exception if the stream is not live.
        if self.readyState != "live":
            raise Exception

        # If the track is video:
        if self.kind == 'video':
            # Checks if a timestamp has been initialized.
            if hasattr(self, "_timestamp"):
                # Commented out:
                    #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE

                # Increments the timestamp by the video packet duration.
                self._timestamp += int(self.VIDEO_PTIME * self.VIDEO_CLOCK_RATE)
                # Increments the frame count.
                self.current_frame_count += 1
                # Calculates the time to wait until the next frame is due to maintain a steady FPS.
                wait = self._start + self.current_frame_count * self.VIDEO_PTIME - time.time()

                # Commented out:
                    # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               

                # If a wait time is needed:
                # Asynchronously waits for the calculated duration.
                if wait > 0:
                    await asyncio.sleep(wait)

                # Commented out:
                    # if len(self.timelist)>=100:
                    #     self.timelist.pop(0)
                    # self.timelist.append(time.time())

            # If this is the first timestamp:
            else:
                # Records the start time.
                self._start = time.time()
                # Initializes the timestamp to 0.
                self._timestamp = 0
                # Appends the start time to the list.
                self.timelist.append(self._start)
                # Logs the start time for debugging.
                logger.info('video start:%f',self._start)

            # Returns the timestamp and the video time base.
            return self._timestamp, self.VIDEO_TIME_BASE

        # If the track is audio:
        else: 
            # Checks if a timestamp has been initialized.
            if hasattr(self, "_timestamp"):
                # Commented out:
                    #self._timestamp = (time.time()-self._start) * SAMPLE_RATE

                # Increments the timestamp by the audio packet duration.
                self._timestamp += int(self.AUDIO_PTIME * self.SAMPLE_RATE)
                # Increments the frame count.
                self.current_frame_count += 1
                # Calculates the time to wait until the next audio frame.
                wait = self._start + self.current_frame_count * self.AUDIO_PTIME - time.time()

                # Commented out:
                    # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                
                # If a wait time is needed:
                if wait > 0:
                    # Asynchronously waits for the calculated duration.
                    await asyncio.sleep(wait)

                # Commented out:
                    # if len(self.timelist)>=200:
                    #     self.timelist.pop(0)
                    #     self.timelist.pop(0)
                    # self.timelist.append(time.time())

            # If this is the first timestamp:
            else:
                # Records the start time.
                self._start = time.time()
                # Initializes the timestamp to 0.
                self._timestamp = 0
                # Appends the start time to the list.
                self.timelist.append(self._start)
                # Logs the audio start time.
                logger.info('audio start:%f',self._start)

            # Returns the timestamp and the audio time base.
            return self._timestamp, self.AUDIO_TIME_BASE

    """
    An asynchronous function that retrieves a frame from the queue. 
    It calls next_timestamp to get the correct timing information, sets the presentation timestamp (PTS) on the frame, and returns it. 
    It also handles end-of-stream conditions and logs performance metrics for video tracks.
    """
    async def recv(self) -> Union[Frame, Packet]:
        # Commented out:
            # frame = self.frames[self.counter % 30]            
        
        # Calls the player's start method to ensure the worker thread is running.
        self._player._start(self)
        # Commented out:
            # if self.kind == 'video':
            #     frame = await self._queue.get()
            # else: #audio
            #     if hasattr(self, "_timestamp"):
            #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
            #         if wait>0:
            #             await asyncio.sleep(wait)
            #         if self._queue.qsize()<1:
            #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
            #             audio = np.zeros((1, 320), dtype=np.int16)
            #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
            #             frame.sample_rate=16000
            #         else:
            #             frame = await self._queue.get()
            #     else:
            #         frame = await self._queue.get()

        # Retrieves a frame and an event point from the asynchronous queue.
        frame, eventpoint = await self._queue.get()
        # Gets the next timestamp and time base for the frame.
        pts, time_base = await self.next_timestamp()
        # Sets the presentation timestamp (pts) of the frame.
        frame.pts = pts
        # Sets the time base of the frame.
        frame.time_base = time_base

        # If an event point is present:
        if eventpoint:
            # Notifies the player with the event point.
            self._player.notify(eventpoint)

        # Checks if the retrieved frame is None, indicating the end of the stream.
        if frame is None:
            # Stops the track.
            self.stop()
            # Raises an exception to stop the recv loop.
            raise Exception

        # If the track is video:
        if self.kind == 'video':
            
            # Adds the time since the last frame to the total time.
            self.totaltime += (time.perf_counter() - self.lasttime)
            # Increments the frame count.
            self.framecount += 1
            # Updates the last time a frame was processed.
            self.lasttime = time.perf_counter()

            # After 100 frames:
            if self.framecount == 100:
                logger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                # Resets the frame count.
                self.framecount = 0
                # Resets the total time.
                self.totaltime=0

        # Returns the processed frame.
        return frame
    
    """
    This method is called to gracefully stop the track, clean up resources, and notify the parent player.
    """
    def stop(self):
        # Calls the parent class's stop method.
        super().stop()
        # If the player object still exists:
        if self._player is not None:
            # Calls the player's stop method for this track.
            self._player._stop(self)
            # Clears the reference to the player.
            self._player = None