import asyncio
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

# Defines the duration of a single audio packet in seconds (20ms).
AUDIO_PTIME = 0.020 
# Specifies the clock rate for the video stream, typically 90 kHz for RTP.
VIDEO_CLOCK_RATE = 90000 
# Defines the duration of a single video packet in seconds (40ms, corresponding to 25fps).
VIDEO_PTIME = 0.040 
# Creates a Fraction object representing the time base for video timestamps.
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE) 
# Sets the sample rate for the audio stream in Hz.
SAMPLE_RATE = 16000 
# Creates a Fraction object representing the time base for audio timestamps.
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE) 

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from logger import logger as mylogger

"""
This class, which inherits from MediaStreamTrack, 
is responsible for managing a single audio or video stream. 
It handles the timing and queueing of frames to be sent over a WebRTC connection.
"""
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
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                # Increments the frame count.
                self.current_frame_count += 1
                # Calculates the time to wait until the next frame is due to maintain a steady FPS.
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()

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
                mylogger.info('video start:%f',self._start)

            # Returns the timestamp and the video time base.
            return self._timestamp, VIDEO_TIME_BASE

        # If the track is audio:
        else: 
            # Checks if a timestamp has been initialized.
            if hasattr(self, "_timestamp"):
                # Commented out:
                    #self._timestamp = (time.time()-self._start) * SAMPLE_RATE

                # Increments the timestamp by the audio packet duration.
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                # Increments the frame count.
                self.current_frame_count += 1
                # Calculates the time to wait until the next audio frame.
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()

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
                mylogger.info('audio start:%f',self._start)

            # Returns the timestamp and the audio time base.
            return self._timestamp, AUDIO_TIME_BASE

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
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
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

"""
This is a target function for a separate thread. 
Its purpose is to run the media rendering loop provided by the container object (likely a live rendering engine like a NeRF model). 
It takes a quit_event to allow the main thread to signal it to stop, an asyncio event loop, the container, and the audio/video tracks.
"""
def player_worker_thread(
    # An event to signal the thread to quit.
    quit_event,
    # The asyncio event loop.
    loop,
     # The container object (nerfreal).
    container,
    # The audio track.
    audio_track,
    # The video track.
    video_track
):
    # Calls the container's render method to start processing media.
    container.render(quit_event,loop,audio_track,video_track)

"""
This class acts as the main interface for a live media source. 
It creates and manages the audio and video PlayerStreamTrack objects and a dedicated worker thread for media processing.
"""
class HumanPlayer:
   
    """
    The constructor creates instances of PlayerStreamTrack for both audio and video and stores a reference to the nerfreal object, 
    which is responsible for generating the media frames.
    """
    def __init__(
        self, nerfreal, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        # A variable to hold the worker thread instance.
        self.__thread: Optional[threading.Thread] = None
        # An event to signal the worker thread to quit.
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        # A set to keep track of started tracks.
        self.__started: Set[PlayerStreamTrack] = set()
        # Variable for the audio track.
        self.__audio: Optional[PlayerStreamTrack] = None
        # Variable for the video track.
        self.__video: Optional[PlayerStreamTrack] = None

        # Initializes the audio track.
        self.__audio = PlayerStreamTrack(self, kind="audio")
        # Initializes the video track.
        self.__video = PlayerStreamTrack(self, kind="video")

        # Stores the nerfreal container object.
        self.__container = nerfreal

    """
    A simple passthrough method that forwards a notification to the underlying nerfreal container.
    """
    def notify(self,eventpoint):
        # Passes the notification to the container object.
        self.__container.notify(eventpoint)

    """
    These provide public access to the audio and video tracks managed by the player.
    """
    # A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
    @property
    def audio(self) -> MediaStreamTrack:
        # Returns the audio track.
        return self.__audio

    # A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
    @property
    def video(self) -> MediaStreamTrack:
        # Returns the video track.
        return self.__video

    """
    This internal method is called by a track when it needs to start. 
    It checks if the worker thread is already running and, 
    if not, initializes and starts a new thread using the player_worker_thread function.
    """
    def _start(self, track: PlayerStreamTrack) -> None:
        # Adds the given track to the set of started tracks.
        self.__started.add(track)
        # If the worker thread hasn't been started yet:
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            # Initializes the quit event.
            self.__thread_quit = threading.Event()
            # Creates a new thread instance.
            self.__thread = threading.Thread(
                # Sets the thread name.
                name="media-player",
                # Sets the function to be executed by the thread.
                target=player_worker_thread,
                # Passes the necessary arguments to the thread function.
                args=(
                    # Passes the quit event as an argument.
                    self.__thread_quit, 
                    # Passes the asyncio event loop.
                    asyncio.get_event_loop(), 
                    # Passes the container object.
                    self.__container, 
                    # Passes the audio track.
                    self.__audio, 
                    # Passes the video track.   
                    self.__video               
                ),
            )

            # Starts the worker thread.
            self.__thread.start()

    """
    This internal method is called when a track stops. 
    If all tracks have stopped, 
    it sets the quit_event to signal the worker thread to shut down and then joins the thread to ensure it has finished.
    """
    def _stop(self, track: PlayerStreamTrack) -> None:
        # Removes the track from the set of started tracks.
        self.__started.discard(track)

        # If no tracks are started and the thread exists:
        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            # Sets the quit event to signal the thread to stop.
            self.__thread_quit.set()
            # Waits for the thread to finish its execution.
            self.__thread.join()
            # Clears the reference to the thread.
            self.__thread = None

        # If no tracks are started and the container exists:
        if not self.__started and self.__container is not None:
            #self.__container.close()

            # Clears the reference to the container.
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        # Logs a debug message using the custom logger.
        mylogger.debug(f"HumanPlayer {msg}", *args)
