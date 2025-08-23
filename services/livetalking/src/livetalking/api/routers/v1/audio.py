from __future__ import annotations

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse

from livetalking.shared.utils import get_settings
from livetalking.shared.models import RecordType

from livetalking.application import RecordApplicationInput
from livetalking.application import AudioTypeApplicationInput
from livetalking.application import IsSpeakingApplicationInput


from logger import get_logger

audio = APIRouter()
logger = get_logger(__name__)

settings = get_settings()

@audio.post(
    '/set_audiotype',
    response_model=None,
)
async def set_audiotype(request: Request, audio_type_input: AudioTypeApplicationInput) -> JSONResponse:
    # Get session ID
    sessionid = audio_type_input.sessionid if audio_type_input.sessionid is not None else 0   

    # Set the audio-type for the BaseReal instance
    request.app.state.nerfreals[sessionid].set_custom_state(audio_type_input.audiotype,audio_type_input.reinit)

    return JSONResponse(
        content={
            'code': 0,
            'data': 'ok'
        },
        headers={"Content-Type": "application/json"}
    )


@audio.post(
    '/record',
    response_model=None,
)
async def record(request: Request, record_input: RecordApplicationInput):
    # Get request parameters
    sessionid = record_input.sessionid if record_input.sessionid is not None else 0

    # Toggle record
    if record_input.type == RecordType.START_RECORD:
        request.app.state.nerfreals[sessionid].start_recording()
    elif record_input.type == RecordType.END_RECORD:
        request.app.state.nerfreals[sessionid].stop_recording()

    # Response success
    return JSONResponse(
        content={
            'code': 0,
            'data': 'ok'
        },
        headers={"Content-Type": "application/json"}
    )


@audio.post(
    '/is_speaking',
    response_model=None
)
async def is_speaking(request: Request, is_speaking_input: IsSpeakingApplicationInput):
    # Get session ID
    sessionid = is_speaking_input.sessionid if is_speaking_input.sessionid is not None else 0

    # Response success
    return JSONResponse(
        content={
            'code': 0,
            "data": request.app.state.nerfreals[sessionid].is_speaking()
        },
        headers={"Content-Type": "application/json"}
    )