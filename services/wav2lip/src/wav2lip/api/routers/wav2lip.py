from __future__ import annotations

from fastapi import Request
from fastapi import APIRouter
from fastapi import Depends

from wav2lip.api.helpers.exception_handler import ExceptionHandler
from wav2lip.application import Wav2lipApplicationInput
from wav2lip.application import Wav2lipApplication

from wav2lip.shared.utils import get_settings
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from authorization import has_authority

from logger import get_logger

wav2lip_router = APIRouter(prefix='/v1')
logger = get_logger(__name__)

settings = get_settings()

@wav2lip_router.post('/wav2lip', dependencies=[Depends(has_authority(authority="PREPARE_AVATAR_DATA"))])
async def query(request: Request, wav2lip_input: Wav2lipApplicationInput) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        wav2lip_service = Wav2lipApplication(
            request=request, 
            settings=settings
        )

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = wav2lip_service.process(
            input=Wav2lipApplicationInput(
                video_url=wav2lip_input.video_url,
                character_id=wav2lip_input.character_id,
            )
        )

    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'video_url': wav2lip_input.video_url})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )