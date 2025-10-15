from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from rag_service.api.helpers.exception_handler import ExceptionHandler
from rag_service.application import RagServiceApplication, RagServiceInput

rag_router = APIRouter(prefix='/v1')
logger = get_logger(__name__)

@rag_router.get('/rag')
async def query(request: Request, rag_input: RagServiceInput, background_tasks: BackgroundTasks) -> JSONResponse:
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    try:
        rag_service = RagServiceApplication(
            request=request
        )
    
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await rag_service.process(
            input=rag_input
        )
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'query': rag_input.query})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )