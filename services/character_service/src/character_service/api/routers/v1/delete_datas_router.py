from __future__ import annotations

from authorization.deps import CurrentToken
from fastapi import APIRouter
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from character_service.api.helpers.exception_handler import ExceptionHandler
from character_service.application.delete_datas_service import DeleteDatasService

from character_service.shared import get_settings
from logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

delete_datas_router = APIRouter()

@delete_datas_router.delete('/users/me/mongo_datas')
def delete_datas_by_user_id(
    request: Request,
    current_token: CurrentToken
    ):
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        delete_datas_service = DeleteDatasService(
            settings=settings,
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )
        
    try:
        response = delete_datas_service.process(user_id=current_token.user_id)
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e))
    
    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )