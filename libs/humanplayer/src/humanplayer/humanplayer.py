from __future__ import annotations

from logger import get_logger

from typing import Optional
from typing import Set
import threading

from .base import PlayerStreamTrack
from aiortc import MediaStreamTrack

import asyncio
from .utils import player_worker_thread

logger = get_logger(__name__)

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

        logger.info("Initializing media tracks...")
        
        # Initializes the audio track.
        self.__audio = PlayerStreamTrack(self, kind="audio")
        logger.info("Audio track initialized")
        
        # Initializes the video track.
        self.__video = PlayerStreamTrack(self, kind="video")
        logger.info(f"Video track initialized: {self.__video.kind}, readyState: {self.__video.readyState}")

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
        logger.info("Getting video track")
        if self.__video is None:
            logger.error("Video track is None!")
        else:
            logger.info(f"Video track status: {self.__video.readyState}")
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
            try:
                self.__thread.start()
                logger.info("Worker thread started successfully")
            except Exception as e:
                logger.error(f"Failed to start worker thread: {str(e)}")
                raise

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
        logger.debug(f"HumanPlayer {msg}", *args)