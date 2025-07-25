# server.py
# from flask import Flask, render_template, send_from_directory, request, jsonify
# from flask_sockets import Sockets
# import base64
import json
# import gevent
# from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
import re
# import numpy as np
from threading import Thread, Event
# import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger

# app = Flask(__name__)
#sockets = Sockets(app)

# Initial variables
nerfreals:Dict[int, BaseReal] = {} # sessionid:BaseReal
opt = None
model = None
avatar = None
        
# webrtc
pcs = set() # Initialize an empty set

# Build a new BaseReal 
# The chosen model base on input model argument when running: python app.py
def build_nerfreal(sessionid:int) -> BaseReal:
    opt.sessionid = sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

# @app.route('/offer', methods=['POST'])
async def offer(request):
    # Define RTC description from request
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Check session limit
    if len(nerfreals) >= opt.max_session:
        logger.info('reach max session')
        return -1

    # Generate a new session ID
    sessionid = len(nerfreals)

    # Build a new BaseReal instance for the session
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal
    
    # Initial new RTCPeerConnection
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Setup event handlers for the peer connection
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]

    # Create a HumanPlayer instance for the session
    # This will handle the audio and video tracks for the session
    # It will also start the worker thread to process the media
    # and notify the NeRFReal instance
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # Set the codec preferences for the video track
    # This will ensure that the video track uses the preferred codecs
    # such as H264, VP8, and RTX
    # This is important for compatibility with different browsers and devices
    # If the browser does not support the preferred codecs, it will fall back to other codecs
    # If no codecs are supported, the video track will not be sent
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    # Set the remote description with the offer
    # This will establish the connection with the remote peer
    # and allow the media tracks to be sent and received
    await pc.setRemoteDescription(offer)

    # Create an SDP answer and set it as the local description
    # This will allow the remote peer to receive the media tracks
    # and establish the connection
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

# input: text
# output: echo or chat response
async def human(request):
    # Get request parameters
    params = await request.json()

    # Get session ID
    sessionid = params.get('sessionid', 0)

    # flush talk if interrupt is set
    if params.get('interrupt'):
        nerfreals[sessionid].flush_talk()

    # response based on type
    if params['type']=='echo':
        nerfreals[sessionid].put_msg_txt(params['text'])
    elif params['type']=='chat':
        res=await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])                         
        nerfreals[sessionid].put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

# input: audio
# output: TODO
async def humanaudio(request):
    try:
        # Asynchronously parse the incoming request's form data.
        form = await request.post()
        # Get the 'sessionid' from the form data. If 'sessionid' is not present, default to 0.
        sessionid = int(form.get('sessionid',0))

        # Access & Read the uploaded file.
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()

        # Processes the audio for the avatar.
        nerfreals[sessionid].put_audio_file(filebytes)

        # Response success
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        # Response fail
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg":"err","data": ""+e.args[0]+""}
            ),
        )

# Set audio type
async def set_audiotype(request):
    # Get request parameters
    params = await request.json()
    # Get session ID
    sessionid = params.get('sessionid',0)    

    # Set the audio-type for the BaseReal instance
    nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

    # Response success
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

# Toggle recording
async def record(request):
    # Get request parameters
    params = await request.json()
    # Get session ID
    sessionid = params.get('sessionid',0)

    # Toggle record
    if params['type']=='start_record':
        nerfreals[sessionid].start_recording()
    elif params['type']=='end_record':
        nerfreals[sessionid].stop_recording()

    # Response success
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

# Check is speaking ?
async def is_speaking(request):
    # Get request parameters
    params = await request.json()
    # Get session ID
    sessionid = params.get('sessionid',0)

    # Response success
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )

# Shutdown handler to close peer connections
# This is called when the application is shutting down
async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

# Post request to a URL with data
# This is used to send the SDP answer to the server
async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data = data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

# Run the NeRFReal instance for a session
# This is called when a new session is created
# It builds the NeRFReal instance and sets up the peer connection
# It also adds the audio and video tracks to the peer connection
# Finally, it sets the local description and sends the SDP answer to the server
async def run(push_url,sessionid):
    # Build a new BaseReal instance for the session
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    # Initial new RTCPeerConnection
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Setup event handlers for the peer connection
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # Create a HumanPlayer instance for the session
    # This will handle the audio and video tracks for the session
    # It will also start the worker thread to process the media
    # and notify the NeRFReal instance
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # Create an SDP offer and set it as the local description
    # This will allow the remote peer to receive the media tracks
    # and establish the connection
    await pc.setLocalDescription(await pc.createOffer())
    # Send the SDP offer 
    answer = await post(push_url,pc.localDescription.sdp)
    # Set the remote description with the answer
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))

####################################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    

if __name__ == '__main__':
    # Set the start method for multiprocessing to 'spawn'. This is important for compatibility
    # and avoiding issues, especially with CUDA, when creating new processes. 'spawn' creates
    # a fresh interpreter process, ensuring that child processes don't inherit unnecessary resources
    # from the parent.
    mp.set_start_method('spawn')

    # Get command-line arguments.
    parser = argparse.ArgumentParser()

    # Arguments related to pose and facial features for the avatar.
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    # A shorthand argument to enable multiple features at once. 'action="store_true"' means it's a flag;
    # if present, its value becomes True.
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    # Arguments related to data handling and workspace.
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options 
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
   
    # Arguments for raymarching in NeRF-like models.
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set (Arguments related to loss functions in training)
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options (Arguments for neural network configuration)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    
    # Arguments for background and eye control.
    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    ### dataset options (Arguments for dataset specifics)
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset) 
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    # Arguments specific to lip handling.
    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    # Arguments for torso-related training.
    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options (Arguments for graphical user interface)
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### Other (Miscellaneous arguments)
    # Arguments for audio processing and embedding.
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    # Arguments for individual codes/embeddings.
    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    # Arguments for ambient and partial data training.
    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    # Arguments for camera training and path smoothing.
    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr (Automatic Speech Recognition) related arguments.
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    # Specific ASR model selection. The commented lines show alternative models.
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto') 
    # parser.add_argument('--asr_model', type=str, default='deepspeech')
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')

    # audio FPS (Frames Per Second for audio processing)
    parser.add_argument('--fps', type=int, default=50)
    # Sliding window left-middle-right length (unit: 20ms) (Parameters for audio feature extraction windowing)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # Arguments for full-body avatar rendering.
    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    # musetalk opt (Arguments specific to the 'musetalk' model)
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # Commented out arguments for custom video settings, indicating they might be deprecated or alternative features.
        # parser.add_argument('--customvideo', action='store_true', help="custom video")
        # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
        # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    # Arguments for Text-to-Speech (TTS) settings.
    parser.add_argument('--tts', type=str, default='edgetts') #xtts gpt-sovits cosyvoice
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000

    # Commented out character and emotion arguments for TTS.
        # parser.add_argument('--CHARACTER', type=str, default='test')
        # parser.add_argument('--EMOTION', type=str, default='default')

    # Main model selection argument.
    parser.add_argument('--model', type=str, default='wav2lip') #musetalk wav2lip

    # Transport protocol selection for streaming.
    parser.add_argument('--transport', type=str, default='webrtc') #rtmp webrtc rtcpush
    # URL for pushing streams, relevant for 'rtmp' or 'rtcpush' transports.
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    # Server configuration arguments.
    parser.add_argument('--max_session', type=int, default=1)  # multi session count (Maximum number of concurrent sessions)
    parser.add_argument('--listenport', type=int, default=8010) # Port for the HTTP server to listen on.

    # Parse the command-line arguments and store them in the 'opt' object.
    opt = parser.parse_args()

    # The following lines are commented out, likely for Flask integration that was replaced by aiohttp.
        #app.config.from_object(opt)
        #print(app.config)
    
    # Initialize 'customopt' list.
    opt.customopt = []
    # If a custom video configuration file is specified, load its JSON content into 'opt.customopt'.
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # Model loading and warm-up section based on the '--model' argument.
    # Each 'elif' block handles a different model type.
    if opt.model == 'ernerf':       
        # Import specific modules for the 'ernerf' model.
        from nerfreal import NeRFReal,load_model,load_avatar
        # Load the model and avatar components for 'ernerf'.
        model = load_model(opt)
        avatar = load_avatar(opt) 
        
        # Commented out section for loading test_loader and initializing multiple NeRFReal instances,
        # indicating previous multi-session setup approach.
            # we still need test_loader to provide audio features for testing.
            # for k in range(opt.max_session):
            #     opt.sessionid=k
            #     nerfreal = NeRFReal(opt, trainer, test_loader,audio_processor,audio_model)
            #     nerfreals.append(nerfreal)
    elif opt.model == 'musetalk':
        # Import specific modules for the 'musetalk' model.
        from musereal import MuseReal,load_model,load_avatar,warm_up
        # Load the model and avatar for 'musetalk'.
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        # Perform warm-up for the 'musetalk' model to optimize performance.
        warm_up(opt.batch_size,model)      
        # Commented out section for initializing multiple MuseReal instances.
            # for k in range(opt.max_session):
            #     opt.sessionid=k
            #     nerfreal = MuseReal(opt,audio_processor,vae, unet, pe,timesteps)
            #     nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip':
        # Import specific modules for the 'wav2lip' model.
        from lipreal import LipReal,load_model,load_avatar,warm_up
        # Load the wav2lip model from the specified path.
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        # Perform warm-up for the wav2lip model.
        warm_up(opt.batch_size,model,256)
        # Commented out section for initializing multiple LipReal instances.
            # for k in range(opt.max_session):
            #     opt.sessionid=k
            #     nerfreal = LipReal(opt,model)
            #     nerfreals.append(nerfreal)
    elif opt.model == 'ultralight':
        # Import specific modules for the 'ultralight' model.
        from lightreal import LightReal,load_model,load_avatar,warm_up
        # Load the model and avatar for 'ultralight'.
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        # Perform warm-up for the 'ultralight' model.
        warm_up(opt.batch_size,avatar,160)

    # If the transport protocol is 'rtmp', initialize a single 'nerfreal' instance and start a rendering thread.
    if opt.transport=='rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # Initialize an aiohttp web application. This will be the main HTTP server.
    appasync = web.Application()
    # Register the 'on_shutdown' coroutine to be called when the application is shutting down.
    # This ensures proper cleanup, like closing WebRTC peer connections.
    appasync.on_shutdown.append(on_shutdown)

    # Define HTTP POST routes and link them to their respective asynchronous handler functions.
    appasync.router.add_post("/offer", offer)           # Handles WebRTC SDP offers.
    appasync.router.add_post("/human", human)           # Handles text input for avatar.
    appasync.router.add_post("/humanaudio", humanaudio) # Handles audio file input for avatar.
    appasync.router.add_post("/set_audiotype", set_audiotype) # Sets audio processing type.
    appasync.router.add_post("/record", record)         # Handles recording commands.
    appasync.router.add_post("/is_speaking", is_speaking) # Checks if the avatar is speaking.
    # Serve static files from the 'web' directory. This is likely where the frontend HTML/JS/CSS resides.
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
        # Allow requests from any origin ("*").
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, # Allow sending cookies and HTTP authentication credentials.
            expose_headers="*",     # Expose all headers to the browser.
            allow_headers="*",      # Allow all headers from the client request.
        )
    })
    # Apply CORS settings to all registered routes in the application.
    for route in list(appasync.router.routes()):
        cors.add(route)

    # Determine the default frontend page name based on the chosen transport protocol.
    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'

    # Log the URL for the http server and the recommended WebRTC frontend.
    # Uses the configured listen port.        
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('If you are using WebRTC, we recommend visiting the WebRTC integrated frontend at:: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')

    # Define a nested function 'run_server' to encapsulate the server startup logic.
    def run_server(runner):
        # Create a new event loop for this thread. This is crucial for running asyncio in a separate thread.
        loop = asyncio.new_event_loop()
        # Set the newly created event loop as the current one for this thread.
        asyncio.set_event_loop(loop)

        # Run the setup phase of the aiohttp application runner within the event loop.
        loop.run_until_complete(runner.setup())
        # Create a TCP site that listens on all available network interfaces ('0.0.0.0') at the specified port.
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        # Start the TCP site within the event loop.
        loop.run_until_complete(site.start())

        # If the transport is 'rtcpush', initiate connections for multiple sessions.
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                # For sessions beyond the first, append the session ID to the push URL.
                if k != 0:
                    push_url = opt.push_url+str(k)
                # Run the 'run' coroutine (which establishes a WebRTC connection and pushes)
                # for each session, within the event loop.
                loop.run_until_complete(run(push_url,k))
        # Start the event loop indefinitely, allowing the server to handle requests.
        loop.run_forever()    
    
    # The following line is commented out, indicating that the server is run directly
    # in the main thread using `run_server(web.AppRunner(appasync))` instead of
    # creating a separate thread for it.
        #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()

    # Run the aiohttp server. This is a blocking call that keeps the script running.
    run_server(web.AppRunner(appasync))

    # The following lines are commented out, likely remnants of a Flask-based setup that was replaced.
        #app.on_shutdown.append(on_shutdown)
        #app.router.add_post("/offer", offer)

        # print('start websocket server')
        # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
        # server.serve_forever()
    
    
