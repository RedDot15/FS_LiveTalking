from __future__ import annotations

from authorization.deps import CurrentToken
from character_service.application.rating_service import AddRatingInput, RatingServiceApplication, UpdateRatingInput
from fastapi import APIRouter
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from character_service.api.helpers.exception_handler import ExceptionHandler

from character_service.shared.utils import get_settings

rating_router = APIRouter()

settings = get_settings()
logger = get_logger(__name__)

@rating_router.get('/ratings')
def list_ratings(request: Request, character_id: str, current_token: CurrentToken) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    if not character_id:
        return exception_handler.handle_bad_request(
            e=f'Character Id is required.',
            extra={}
        )

    try:
        rating_service = RatingServiceApplication(
            settings=settings,
            request=request
        )
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = rating_service.process(character_id=character_id)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@rating_router.post('/ratings')
def create_rating(request: Request, body: AddRatingInput, current_token: CurrentToken) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        rating_service = RatingServiceApplication(
            settings=settings,
            request=request
        )
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = rating_service.add_rating(
            input=body,
            current_user_id=current_token.id,
            current_username=current_token.username)
        
    except Exception as e:
        return exception_handler.handle_exception(
            e=str(e),
            extra={}
        )

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@rating_router.put('/ratings/{rating_id}')
def update_rating(request: Request, rating_id: str, body: UpdateRatingInput, current_token: CurrentToken) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        rating_service = RatingServiceApplication(
            settings=settings,
            request=request
        )
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        body.rating_id = rating_id
        response = rating_service.update_rating(
            input=body,
            current_user_id=current_token.id,
            current_username=current_token.username)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@rating_router.delete('/ratings/{rating_id}')
def delete_rating(request: Request, rating_id: str, current_token: CurrentToken) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        rating_service = RatingServiceApplication(
            settings=settings,
            request=request
        )
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = rating_service.delete_rating(
            rating_id=rating_id,
            current_user_id=current_token.id,
            current_username=current_token.username)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )