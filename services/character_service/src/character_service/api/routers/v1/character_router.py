from __future__ import annotations

from authorization.deps import CurrentToken
from fastapi import APIRouter
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from character_service.api.helpers.exception_handler import ExceptionHandler
from character_service.application import CharacterServiceApplication

from character_service.shared.utils import get_settings

character_router = APIRouter()

settings = get_settings()
logger = get_logger(__name__)

@character_router.get('/characters')
def list_characters(request: Request, current_token: CurrentToken) -> JSONResponse:

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
        response = character_service.process()
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@character_router.get('/users/me/characters')
def list_my_characters(request: Request, current_token: CurrentToken) -> JSONResponse:

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
        response = character_service.get_by_created_by(created_by=current_token.id)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@character_router.delete('/characters/{character_id}')
async def delete_character(request: Request, current_token: CurrentToken, character_id: str) -> JSONResponse:

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
        response = await character_service.delete_character(character_id=character_id, user_id=current_token.id, scope=current_token.scope)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )