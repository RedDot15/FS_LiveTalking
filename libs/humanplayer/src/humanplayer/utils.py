from __future__ import annotations

from logger import get_logger

logger = get_logger(__name__)


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
