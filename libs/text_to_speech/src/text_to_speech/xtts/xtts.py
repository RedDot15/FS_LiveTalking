from __future__ import annotations

import time
import resampy
import numpy as np
from typing import Iterator

import requests

from .settings import XTTSSettings

from ..base import BaseTTS
from logger import get_logger

logger = get_logger(__name__)

class XTTS(BaseTTS):
    
    def __init__(self, parent, settings: XTTSSettings):
        super().__init__(parent=parent)
        
        self.settings = settings
        self.speaker = self.get_speaker(self.settings.ref_file)
    
    def txt_to_audio(self,msg):
        # Use XTTS to convert text to audio
        text, textevent = msg  

         # Start streaming the audio from the XTTS service
        self.stream_tts(
             # Call the xtts method to get a generator for audio chunks
            self.xtts(
                text,
                self.speaker,
                "vi", #en args.language,  # Set the language to Chinese (zh-cn)
                self.settings.TTS_SERVER, # The URL of the TTS server
                "20" #args.stream_chunk_size  # The size of each audio chunk in milliseconds
            ),
            msg
        )
        
    # def get_speaker(self,ref_audio,server_url):
        # Clone a speaker from a reference audio file
        # Prepare the files to be sent in the POST request
        # files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}

        # Send a POST request to the server's '/clone_speaker' endpoint
        # response = requests.post(f"{server_url}/clone_speaker", files=files)

        # Return the JSON response, which contains the speaker's information
        # return response.json()

    def get_speaker(self, ref_audio):
        return {'speaker_wav': ref_audio}

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        # Generate streaming audio from text using the XTTS service
        start = time.perf_counter()
        # Add the text, language, and chunk size to the speaker dictionary for the request payload
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            # Send a GET request to the server's '/tts_stream' endpoint
            res = requests.get(
                f"{server_url}/tts_stream",
                params=speaker,
            )

            # Log the time it took to make the POST request
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            # Check if the request was successful
            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            # Iterate over the content of the streaming response in chunks
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                # Check if it's the first chunk received
                if first:
                    end = time.perf_counter()
                    # Log the time it took to receive the first chunk
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False

                # If a chunk is received and it's not empty, yield it
                if chunk:
                    yield chunk
                    
        except Exception as e:
            print(e)
    
    def stream_tts(self, audio_stream, msg):
        # Process the incoming audio stream from the TTS service
        text, textevent = msg
        first = True
        # Loop through each audio chunk received from the generator
        for chunk in audio_stream:
            # Check if the chunk is not None and has a length greater than 0
            if chunk is not None and len(chunk) > 0:          
                # Convert the raw byte chunk (16-bit integers) to a normalized float32 numpy array
                stream = np.frombuffer(chunk, dtype = np.int16).astype(np.float32) / 32767
                # Resample the audio from its original sample rate (24000 Hz for XTTS) to the target sample rate (16000 Hz)
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)

                # Commented out:
                    # byte_stream = BytesIO(buffer)
                    # stream = self.__create_bytes_stream(byte_stream)

                streamlen = stream.shape[0]
                idx = 0
                
                # Loop through the resampled audio stream and extract chunks of the specified size
                while streamlen >= self.chunk:
                    eventpoint=None
                    # If it's the first frame, create a 'start' event point
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    
                    # Put the audio frame and its event point into the parent's audio queue
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk

        # After all chunks are processed, create an 'end' event point
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        # Put a final silent audio frame with the 'end' event point into the parent's audio queue
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 