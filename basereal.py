# The primary purposes of this file are:
    # 1. Real-time Text-to-Speech (TTS) Integration: It provides an interface to various TTS engines (EdgeTTS, GPT-Sovits, XTTS, CosyVoiceTTS, FishTTS, TencentTTS), allowing the system to synthesize speech from text input.
    # 2. Audio Processing and ASR Integration: It handles the input of audio data, including file-based audio, converting it to a standardized format (16kHz, mono, float32) and chunking it for an (unseen in this snippet) Automatic Speech Recognition (ASR) component.
    # 3. Video and Audio Recording: It enables the recording of real-time video frames and audio chunks into separate temporary files using ffmpeg subprocesses, and then combines them into a final MP4 video file.
    # 4. Custom Animation and Audio Synchronization: It supports loading and managing "custom" animation cycles (image sequences) synchronized with corresponding audio. This allows for pre-defined visual and auditory responses or behaviors, potentially triggered by certain events or states.

import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

# Imports the PyAV library for multimedia processing.
import av
# Imports Fraction for representing rational numbers.
from fractions import Fraction

from ttsreal import EdgeTTS, SovitsTTS, XTTS, CosyVoiceTTS, FishTTS, TencentTTS
from logger import logger

# Imports tqdm for displaying progress bars.
from tqdm import tqdm

# Reads a list of image paths and returns them as a list of OpenCV image objects.
def read_imgs(img_list):
    frames = []
    # Iterates through image paths with a progress bar.
    for img_path in tqdm(img_list):
        # Reads an image using OpenCV.
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# Base class for real-time processing, handling TTS, audio/video recording, and custom animation cycles.
class BaseReal:
    def __init__(self, opt):
        ## Sets the options for the BaseReal instance.
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps
        self.sessionid = self.opt.sessionid

        # Initializes the appropriate Text-to-Speech (TTS) engine based on the 'opt.tts' setting.
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        
        # Initializes a flag to indicate if speech is active.
        self.speaking = False

        # Initializes a flag to indicate if recording is active.
        self.recording = False
        # Placeholder for the video recording subprocess pipe.
        self._record_video_pipe = None
        # Placeholder for the audio recording subprocess pipe.
        self._record_audio_pipe = None
        # Initializes video width and height.
        self.width = self.height = 0

        
        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        # Load custom assets.
        self.__loadcustom()

    # Sends a text message to the TTS engine for synthesis.
    def put_msg_txt(self, msg, eventpoint = None):
        self.tts.put_msg_txt(msg, eventpoint)
    
    # Sends an audio chunk to the ASR (Automatic Speech Recognition) engine.
    def put_audio_frame(self, audio_chunk, eventpoint=None): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk, eventpoint)

    # Processes a byte stream representing an audio file and feeds it to the ASR in chunks.
    def put_audio_file(self, filebyte): 
         # Creates an in-memory byte stream.
        input_stream = BytesIO(filebyte)
        # Processes the byte stream into an audio array.
        stream = self.__create_bytes_stream(input_stream)
        # Gets the total length of the audio stream.
        streamlen = stream.shape[0]
        # Initializes index for chunking.
        idx = 0
        # Loops while there are enough samples for a chunk.
        while streamlen >= self.chunk:  #and self.state == State.RUNNING
            # Puts the current audio chunk for processing.
            self.put_audio_frame(stream[idx:idx+self.chunk])
            # Decrements the remaining stream length.
            streamlen -= self.chunk
            # Increments the index.
            idx += self.chunk
    
    # Internal method to read an audio byte stream, convert it to float32,
    # handle multiple channels, and resample if necessary.
    def __create_bytes_stream(self, byte_stream):
        # Reads audio data and its sample rate.
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        # Logs the sample rate and shape of the audio stream.
        logger.info(f'[INFO] Put audio stream {sample_rate}: {stream.shape}')
        # Converts the stream to float32 data type.
        stream = stream.astype(np.float32)

        # Warns if audio has multiple channels.
        if stream.ndim > 1:
            logger.info(f'[WARN] Audio has {stream.shape[1]} channels, only use the first.')
            # Selects only the first channel.
            stream = stream[:, 0]
    
        # Warns and resamples if sample rates differ.
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] Audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x = stream, sr_orig = sample_rate, sr_new = self.sample_rate)

        # Returns the processed audio stream.
        return stream

    # Flushes any pending speech or ASR processing.
    def flush_talk(self):
        # Flushes the TTS engine.
        self.tts.flush_talk()
        # Flushes the ASR engine.
        self.asr.flush_talk()

    # Checks if the system is currently speaking.
    def is_speaking(self) -> bool:
        return self.speaking
    
    # Loads custom image and audio cycles based on the provided options.
    def __loadcustom(self):
        for item in self.opt.customopt:
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

    # Resets the current state and all custom audio/image indices.
    def init_customindex(self):
        # Resets the current state.
        self.curr_state=0
        for key in self.custom_audio_index:
            # Resets each custom audio index.
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            # Resets each custom index.
            self.custom_index[key]=0

    # Logs a notification message with an event point.
    def notify(self, eventpoint):
        logger.info("notify: %s", eventpoint)

    #Starts recording video and audio using FFmpeg subprocesses.
    def start_recording(self):
        # Checks if recording is already active.
        if self.recording:
            return

        # FFmpeg command for video recording (raw video input to h264 mp4 output).
        command = ['ffmpeg',
                    '-y', '-an',                                   # Overwrite output files without asking, no audio.
                    '-f', 'rawvideo',                              # Input format is raw video.
                    '-vcodec','rawvideo',                          # Video codec for input is raw video.
                    '-pix_fmt', 'bgr24',                           # Input pixel format is BGR24.
                    '-s', "{}x{}".format(self.width, self.height), # Video resolution.
                    '-r', str(25),                                 # Frame rate.
                    '-i', '-',                                     # Input from stdin.
                    '-pix_fmt', 'yuv420p',                         # Output pixel format.
                    '-vcodec', "h264",                             # Output video codec.
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']               # Output video file name.
        # Starts the video recording subprocess.
        self._record_video_pipe = subprocess.Popen(command, shell = False, stdin = subprocess.PIPE)

        # FFmpeg command for audio recording (raw audio input to aac output).
        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        # Starts the audio recording subprocess.
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    # Writes video frame data to the recording pipe.
    def record_video_data(self, image):
        # Checks if video dimensions are not yet set.
        if self.width == 0:
            print("image.shape:", image.shape)
            # Sets height and width from the image shape.
            self.height, self.width,_ = image.shape
        # Checks if recording is active.
        if self.recording:
            # Writes the image data to the video pipe.
            self._record_video_pipe.stdin.write(image.tostring())

    # Writes audio frame data to the recording pipe.
    def record_audio_data(self, frame):
        # Checks if recording is active.
        if self.recording:
            # Writes the audio data to the audio pipe.
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    # Stops the recording process, closes FFmpeg pipes, and combines audio and video.
    def stop_recording(self):
        # Checks if recording is not active.
        if not self.recording:
            return
        # Sets the recording flag to False.
        self.recording = False 
        # Closes the video pipe's stdin.
        self._record_video_pipe.stdin.close()  
        # Waits for the video recording process to finish.
        self._record_video_pipe.wait()
        # Closes the audio pipe's stdin.
        self._record_audio_pipe.stdin.close()
        # Waits for the audio recording process to finish.
        self._record_audio_pipe.wait()
        # FFmpeg command to combine the recorded audio and video into a single MP4 file.
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        # Executes the FFmpeg command.
        os.system(cmd_combine_audio) 
        # Commented out: likely for removing temporary files.
            #os.remove(output_path)

    # Calculates a mirrored index within a given size, creating a back-and-forth cycle.
    def mirror_index(self, size, index):
        # Calculates how many full turns have occurred.
        turn = index // size
        # Calculates the remainder within the current turn.
        res = index % size
        # If it's an even turn (forward direction).
        if turn % 2 == 0:
            return res
        # If it's an odd turn (backward direction).
        else:
            return size - res - 1 
    
    # Retrieves a chunk of audio from the custom audio cycle for a given audio type.
    def get_audio_stream(self, audiotype):
        # Gets the current audio index for the type.
        idx = self.custom_audio_index[audiotype]
        # Extracts an audio chunk.
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        # Increments the audio index.
        self.custom_audio_index[audiotype] += self.chunk
        # Checks if the end of the custom audio cycle has been reached.
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  # The current video does not loop, switch to silent state.
        # Returns the audio chunk.
        return stream
    
    # Sets the current custom state and optionally reinitializes the custom audio and image indices.
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        # Sets the current state.
        self.curr_state = audiotype
        # Check if reinitialization is requested.
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
    
    # def process_custom(self, audiotype:int, idx:int):
    #     if self.curr_state != audiotype: # Switching from inference to lip-sync
    #         if idx in self.switch_pos:  # Can switch at a keyframe position
    #             self.curr_state = audiotype
    #             self.custom_index = 0
    #     else:
    #         self.custom_index +=1