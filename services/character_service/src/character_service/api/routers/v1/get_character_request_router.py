from __future__ import annotations

from authorization.deps import CurrentToken
from fastapi import APIRouter
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from character_service.api.helpers.exception_handler import ExceptionHandler
from character_service.application.get_request_character import GetRequestCharacterApplication

from character_service.shared import get_settings
from logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

get_character_request_router = APIRouter()

@get_character_request_router.get('/requests')
def get_character_request(
    request: Request,
    current_token: CurrentToken
    ):
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        request_character_service = GetRequestCharacterApplication(
            settings=settings,
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )
        
    try:
        response = request_character_service.process()
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))
    
    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )