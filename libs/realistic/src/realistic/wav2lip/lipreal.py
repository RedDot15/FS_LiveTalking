from __future__ import annotations

import asyncio
import time

import numpy as np
import cv2
import copy
import torch
import torch.multiprocessing as mp

import json

import queue
from queue import Queue
from threading import Thread

from ..base import BaseReal
from av import AudioFrame, VideoFrame

from asr import LipASR

from logger import get_logger

from .utils import inference
from .settings import LipRealSettings

from typing import Any

import glob
import os
from .utils import read_imgs
from .utils import load_avatar

import soundfile as sf

from text_to_speech import XTTS
from queue import Queue

logger = get_logger(__name__)

class LipReal(BaseReal):
    
    def __init__(self, avatar: tuple[list, list, Any], model: Any, character_id: str, sessionid: str, settings: LipRealSettings) -> None:
        super().__init__(sessionid, character_id)
        
        self.settings = settings
        
        self.W = settings.W
        
        self.H = settings.H
        
        self.fps = settings.fps
        
        self.batch_size = settings.batch_size
        
        self.idx = settings.idx
        
        self.res_frame_queue: Queue = Queue(maxsize=settings.batch_size*2)
        
        self.model = model
        
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        
        self.asr = LipASR(self)
        self.tts = XTTS(self, character_id=character_id, settings=settings.xtts)
        
        self.asr.warm_up()        
        
        self.render_event = mp.Event()
        
        self.__loadcustom()

    # Destructor method, logs a message when the object is deleted.
    def __del__(self):
        logger.info(f'lipreal delete')

    # Thread target function to retrieve generated frames from the queue, 
    # blend them with the full avatar image, and send them to the output tracks.
    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):

        # Loops until the quit event is set.
        while not quit_event.is_set():
            try:
                # Gets a result tuple from the queue with a timeout.
                res_frame, idx, audio_frames = self.res_frame_queue.get(block = True, timeout = 1)
            except queue.Empty:
                # If the queue is empty, continue to the next iteration.
                continue

            # Checks if all audio frames in the batch are from a silent/custom source (type != 0).
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0: # All are silent, just use the full image.
                # Sets the speaking flag to False.
                self.speaking = False
                # Gets the audio type.
                audiotype = audio_frames[0][1]

                # Checks if there is a custom video for this audio type.
                if self.custom_index.get(audiotype) is not None: 
                    # Gets a mirrored index for the custom video.
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.custom_index[audiotype])
                    # Gets the custom image frame.
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    # Increments the custom image index.
                    self.custom_index[audiotype] += 1

                    # Commented out: current video does not loop, switch to silent state
                        # if not self.custom_opt[audiotype].loop and self.custom_index[audiotype] >= len(self.custom_img_cycle[audiotype]):
                        #   self.curr_state = 1  
                else:
                    # Gets the full image frame based on the index.
                    combine_frame = self.frame_list_cycle[idx]

                    # commented out:
                        #combine_frame = self.imagecache.get_img(idx)
            else:
                # Sets the speaking flag to True.
                self.speaking = True
                # Gets the bounding box for the face from the coordinates list.
                bbox = self.coord_list_cycle[idx]
                # Creates a deep copy of the full image frame.
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                #combine_frame = copy.deepcopy(self.imagecache.get_img(idx))

                # Unpacks the bounding box coordinates.
                y1, y2, x1, x2 = bbox
                try:
                    # Resizes the generated lip-synced face frame to fit the bounding box.
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    # Skips the frame if resizing fails.
                    continue

                # commented out:
                    # combine_frame = get_image(ori_frame,res_frame,bbox)
                    # t=time.perf_counter()
                
                # Blends the generated face frame onto the full image.
                combine_frame[y1:y2, x1:x2] = res_frame
                
                # commented out:
                    # print('blending time:',time.perf_counter()-t)

            image = combine_frame #(outputs['image'] * 255).astype(np.uint8)
            
            # Creates a PyAV video frame.
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            # Puts the new video frame into the video track's queue using a thread-safe call.
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            # Records the video frame.
            self.record_video_data(image)

            # Processes each audio frame in the batch.
            for audio_frame in audio_frames:
                # Unpacks the audio data and metadata.
                frame, type, eventpoint = audio_frame
                # Scales and converts the audio frame to 16-bit integer PCM.
                frame = (frame * 32767).astype(np.int16)
                # Creates a PyAV audio frame.
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                # Fills the audio frame with byte data.
                new_frame.planes[0].update(frame.tobytes())
                # Sets the sample rate.
                new_frame.sample_rate = 16000

                # Commented out:
                    # if audio_track._queue.qsize()>10:
                    #     time.sleep(0.1)

                # Puts the new audio frame into the audio track's queue using a thread-safe call.
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                # Records the audio frame.
                self.record_audio_data(frame)

                # Commented out: notify eventpoint
                    #self.notify(eventpoint)

        logger.info('lipreal process_frames thread stop') 
            
    # The main rendering loop, which starts the TTS, ASR, inference, and frame processing threads.
    def render(self, quit_event, loop = None, audio_track = None, video_track = None):
        # Commented out:
            # if self.opt.asr:
            #   self.asr.warm_up()

        # Starts the TTS rendering process.
        self.tts.render(quit_event)
        # Initializes custom animation indices.
        self.init_customindex()
        # Starts a thread for processing the generated frames.
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        # Starts a thread for the Wav2Lip inference.
        Thread(target = inference, args = (quit_event,self.batch_size,self.face_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.model,)).start()  #mp.Process

        # commented out event set:
            # self.render_event.set() #start infer process render

        count = 0
        totaltime = 0
        _starttime = time.perf_counter()
        
        # commented out:
        #   _totalframe=0

        # Loops until the quit event is set.
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()

            # Runs a step of the ASR process.
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)

            # Implements a delay if the video output queue is getting too full.
            if video_track._queue.qsize() >= 5:
                logger.debug('sleep qsize=%d', video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)

            # Commented out:    
                # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
                # if delay > 0:
                #     time.sleep(delay)

        # commented out event clear:
        #   self.render_event.clear() #end infer process render
        
        logger.info('lipreal thread stop')
            
    # Flushes any pending speech or ASR processing.
    def flush_talk(self):
        # Flushes the TTS engine.
        self.tts.flush_talk()
        # Flushes the ASR engine.
        self.asr.flush_talk()

    # Loads custom image and audio cycles based on the provided options.
    def __loadcustom(self):
        
        customopt = []
        
        if self.settings.customvideo_config != '':
            with open(self.settings.customvideo_config,'r') as file:
                customopt = json.load(file)
                
        for item in customopt:
            logger.info(item)
            # Gathers image files from the specified path, sorted numerically.
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key = lambda x : int(os.path.splitext(os.path.basename(x))[0]))
            # Reads images and stores them in the custom image cycle dictionary.
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            # Reads custom audio and its sample rate.
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            # Initializes audio index for the custom type.
            self.custom_audio_index[item['audiotype']] = 0
            # Initializes general index for the custom type.
            self.custom_index[item['audiotype']] = 0
            # Stores custom options for the type.
            self.custom_opt[item['audiotype']] = item
