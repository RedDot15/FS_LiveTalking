
from __future__ import annotations

import json
import asyncio

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from authorization.deps import CurrentToken

# from chat_service.application import ChatServiceInput
# from chat_service.application import ChatServiceApplication
from livetalking.api.helpers.exception_handler import ExceptionHandler
from livetalking.shared.utils import get_settings

from livetalking.shared.tools import build_nerfreal
from livetalking.application import OfferApplicationInput
from livetalking.application import OfferApplication

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender

from humanplayer import HumanPlayer
from realistic import load_avatar

from logger import get_logger

offer_router = APIRouter()
logger = get_logger(__name__)

settings = get_settings()

@offer_router.post(
    '/offer',
    response_model=None,
)
async def offer(request: Request, offer_input: OfferApplicationInput, current_token: CurrentToken) -> JSONResponse:
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    logger.info("Received offer request")
    
    # Check session limit
    if len(request.app.state.nerfreals) >= request.app.state.settings.max_session:
        logger.info('Reached max session limit')
        return exception_handler.handle_exception(
            e='Maximum session limit reached',
            extra={},
        )
        
    try:
        offer_application = OfferApplication(
            request=request,
            settings=settings
        )
        
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )
    try:
        response_data = await offer_application.process(
            input=OfferApplicationInput(
                sdp=offer_input.sdp,
                type=offer_input.type,
                character_id=offer_input.character_id,
                session_id=current_token.id
            )
        )
        
    except Exception as e:
        return exception_handler.handle_exception(
            e=str(e), 
            extra={
                'character_id': offer_input.character_id
            }
        )

    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(response_data)
    )
