from __future__ import annotations
import os

from base import BaseModel
from base import BaseService

from typing import Annotated
from typing import Any

from pydantic import ConfigDict
from pydantic import Field

from realistic import LipReal
from realistic import load_avatar

from aiortc import RTCPeerConnection
from aiortc import RTCSessionDescription

from humanplayer import HumanPlayer
from aiortc.rtcrtpsender import RTCRtpSender
from livetalking.shared.tools import build_nerfreal

from logger import get_logger

logger = get_logger(__name__)

class OfferApplicationInput(BaseModel):
    sdp: str
    type: str
    character_name: str
    
class OfferApplicationOutput(BaseModel):
    sdp: str
    type: str
    sessionid: int

class OfferApplication(BaseService):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: Annotated[Any, Field(exclude=True)]
    settings: Annotated[Any, Field(exclude=True)]
    
    async def process(self, input: OfferApplicationInput) -> OfferApplicationOutput:
        
        offer = RTCSessionDescription(
            sdp=input.sdp, 
            type=input.type
        )
        
        logger.info("Created RTCSessionDescription from offer")
        
        sessionid = len(self.request.app.state.nerfreals)
        logger.info(f"Generated new session ID: {sessionid}")
        
        self.request.app.state.nerfreals[sessionid] = None
        
        full_imgs_path, face_imgs_path, coords_path = self._ensure_avatars(
            character_name=input.character_name
        )

        try:
            nerfreal = await build_nerfreal(
                nerfreals=self.request.app.state.nerfreals,
                avatar=load_avatar(
                    full_imgs_path=full_imgs_path,
                    face_imgs_path=face_imgs_path,
                    coords_path=coords_path
                ),
                model=self.request.app.state.model,
                character_name=input.character_name,
                sessionid=sessionid
            )
            
        except Exception as e:
            logger.error(f"Error building NerfReal instance: {str(e)}")
            self.request.app.state.nerfreals.pop(sessionid, None)
            raise e
        
        self.request.app.state.nerfreals[sessionid] = nerfreal
        logger.info(f"NerfReal instance built successfully for session {sessionid}")

        pc = RTCPeerConnection()
        self.request.app.state.pcs.add(pc)
        logger.info("Created new RTCPeerConnection")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed to: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.request.app.state.pcs.discard(pc)
                del self.request.app.state.nerfreals[sessionid]

        player = HumanPlayer(self.request.app.state.nerfreals[sessionid])
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


        logger.info(f"Sending WebRTC answer")

        return OfferApplicationOutput(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
            sessionid=sessionid
        )
        
    def _ensure_avatars(self, character_name: str) -> None:
        
        avatar_path = f"./data/avatars/{character_name}"
        
        if not os.path.exists(avatar_path):
            os.makedirs('./data/avatars', exist_ok=True)
            self.request.app.state.minio_client.get_folder(
                bucket_name=self.settings.bucket_name,
                des_folder_name=character_name,
                prefix='avatars',
                local_folder_path='data'
            )

        full_imgs_path = f"./data/{character_name}/avatars/full_imgs"
        # Constructs the path for face images.
        face_imgs_path = f"./data/{character_name}/avatars/face_imgs"
        # Constructs the path for coordinates file.
        coords_path = f"./data/{character_name}/avatars/coords.pkl"
        
        return full_imgs_path, face_imgs_path, coords_path
        