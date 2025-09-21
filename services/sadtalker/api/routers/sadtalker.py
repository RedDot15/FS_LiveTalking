from __future__ import annotations


from api.helpers.exception_handler import ExceptionHandler
from fastapi import APIRouter
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from application import SadTalkerServiceInput
from application import SadTalkerService
from shared.logger import get_logger

from shared.utils import get_settings

settings = get_settings()
logger = get_logger(__name__)

sadtalker_router = APIRouter(prefix='/v1')

@sadtalker_router.post('/sadtalker')
def sadtalker(request: Request, sadtalker_input: SadTalkerServiceInput) -> JSONResponse:
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        sadtalker_service = SadTalkerService(
            request=request, 
            settings=settings
        )

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = sadtalker_service.process(
            input=SadTalkerServiceInput(
                bucket_name=sadtalker_input.bucket_name,
                character_name=sadtalker_input.character_name,
                image_url=sadtalker_input.image_url,
                audio_url=sadtalker_input.audio_url
            )
        )
    except Exception as e:
        return exception_handler.handle_exception(
            e=str(e), 
            extra={
                'image_url': sadtalker_input.image_url,
                "audio_url": sadtalker_input.audio_url
            }
        )

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )