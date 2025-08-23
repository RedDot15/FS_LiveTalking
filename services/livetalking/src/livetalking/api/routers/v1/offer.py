
from __future__ import annotations

import json
import asyncio

from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# from chat_service.application import ChatServiceInput
# from chat_service.application import ChatServiceApplication
from livetalking.api.helpers.exception_handler import ExceptionHandler
from livetalking.shared.utils import get_settings

from livetalking.application import LiveTalkingApplication
from livetalking.shared.tools import build_nerfreal
from livetalking.application import OfferApplicationInput
from livetalking.application import OfferApplicationOutput

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender

from humanplayer import HumanPlayer

from logger import get_logger

offer_router = APIRouter()
logger = get_logger(__name__)

settings = get_settings()

@offer_router.post(
    '/offer',
    response_model=None,
)
async def offer(request: Request, offer_input: OfferApplicationInput) -> JSONResponse:
    try:
        logger.info("Received offer request")
        
        # Start WebRTC negotiation process
        offer = RTCSessionDescription(sdp=offer_input.sdp, type=offer_input.type)
        logger.info("Created RTCSessionDescription from offer")

        # Check session limit
        if len(request.app.state.nerfreals) >= request.app.state.settings.max_session:
            logger.info('Reached max session limit')
            return JSONResponse(
                status_code=503,
                content={"error": "Maximum session limit reached"}
            )

        # Generate session ID and initialize
        sessionid = len(request.app.state.nerfreals)
        logger.info(f"Generated new session ID: {sessionid}")
        
        # Build NerfReal instance
        request.app.state.nerfreals[sessionid] = None
        nerfreal = await build_nerfreal(
            nerfreals=request.app.state.nerfreals,
            avatar=request.app.state.avatar,
            model=request.app.state.model,
            sessionid=sessionid
        )
        
        if nerfreal is None:
            raise Exception("Failed to build NerfReal instance")
        
        request.app.state.nerfreals[sessionid] = nerfreal
        logger.info(f"NerfReal instance built successfully for session {sessionid}")

        # Setup WebRTC connection
        pc = RTCPeerConnection()
        request.app.state.pcs.add(pc)
        logger.info("Created new RTCPeerConnection")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed to: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                request.app.state.pcs.discard(pc)
                del request.app.state.nerfreals[sessionid]

        # Setup media player and tracks
        player = HumanPlayer(request.app.state.nerfreals[sessionid])
        logger.info("Created HumanPlayer instance")
        
        audio_sender = pc.addTrack(player.audio)
        video_sender = pc.addTrack(player.video)
        
        logger.info("Added audio and video tracks")

        # Set video codec preferences
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name in ["H264", "VP8", "rtx"], capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)
        logger.info("Set video codec preferences")

        # Complete WebRTC setup
        await pc.setRemoteDescription(offer)
        logger.info("Set remote description")
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        logger.info("Created and set local description")

        # Prepare response
        response_data = {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "sessionid": sessionid
        }
        logger.info(f"Sending WebRTC answer")
        
        return JSONResponse(
            content=response_data,
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logger.error(f"Error in offer handler: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )