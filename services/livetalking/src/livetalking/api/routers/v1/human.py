from __future__ import annotations

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from livetalking.shared.utils import get_settings

from livetalking.application import HumanApplicationInput
from livetalking.shared.models import HumanType
from livetalking.shared.tools import llm_response

import asyncio

from logger import get_logger

human_router = APIRouter()
logger = get_logger(__name__)

settings = get_settings()

@human_router.post(
    '/human',
    response_model=None,
)
async def human(request: Request, human_input: HumanApplicationInput) -> JSONResponse:
    # Get request parameters
    # Get session ID
    sessionid = human_input.sessionid if human_input.sessionid is not None else 0

    # # flush talk if interrupt is set
    # if human_input.interrupt is not None:
    #     request.app.state.nerfreals[sessionid].flush_talk()

    # response based on type
    if human_input.type == HumanType.ECHO:
        logger.info('ECHO message received, sending back the same text.')
        request.app.state.nerfreals[sessionid].put_msg_txt(human_input.text)
        
    elif human_input.type == HumanType.CHAT:
        logger.info('CHAT message received, processing with LLM.')
        await llm_response(nerfreal=request.app.state.nerfreals[sessionid], text=human_input.text)

    return JSONResponse(
        content={
            'code': 0,
            'data': 'ok'
        },
        headers={"Content-Type": "application/json"}
    )