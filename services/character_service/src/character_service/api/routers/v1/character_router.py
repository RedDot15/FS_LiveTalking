from __future__ import annotations

from fastapi import APIRouter
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from character_service.api.helpers.exception_handler import ExceptionHandler
from character_service.application import CharacterServiceApplication

from character_service.shared.utils import get_settings

character_router = APIRouter(prefix='/v1')

settings = get_settings()
logger = get_logger(__name__)

@character_router.get('/characters')
async def list_characters(request: Request) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        character_service = CharacterServiceApplication(
            settings=settings,
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await character_service.process()
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )