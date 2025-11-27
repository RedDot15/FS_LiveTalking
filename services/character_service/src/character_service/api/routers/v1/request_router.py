from __future__ import annotations
from typing import Annotated

from authorization.deps import CurrentToken, has_authority
from authorization.model import TokenPayload
from character_service.application.request_service import RequestInput, RequestRejectInput
from fastapi import APIRouter, Depends
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from character_service.api.helpers.exception_handler import ExceptionHandler
from character_service.application.request_service import RequestServiceApplication

from character_service.shared.utils import get_settings

character_router = APIRouter()

settings = get_settings()
logger = get_logger(__name__)


@character_router.post('/requests')
async def add_creation_request(request: Request, body: RequestInput, current_token: CurrentToken) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        request_service = RequestServiceApplication(
            settings=settings,
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await request_service.add_creation_request(input=body, current_user_id=current_token.id)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@character_router.patch('/requests/{request_id}/approve')
async def approve_request(request: Request, request_id: str, permitted_token: Annotated[TokenPayload, Depends(has_authority(authority="APPROVE_REQUEST"))]) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        request_service = RequestServiceApplication(
            settings=settings,
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        await request_service.approve_request(request_id=request_id, current_user_id=permitted_token.id)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            "APPROVED",
        )
    )

@character_router.patch('/requests/{request_id}/reject')
async def reject_request(request: Request, request_id: str, body: RequestRejectInput, permitted_token: Annotated[TokenPayload, Depends(has_authority(authority="REJECT_REQUEST"))]) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        request_service = RequestServiceApplication(
            settings=settings,
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        body.request_id = request_id
        response = await request_service.reject_request(inputs=body, current_user_id=permitted_token.id)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))

    return exception_handler.handle_success(
        jsonable_encoder(
            "REJECTED",
        )
    )