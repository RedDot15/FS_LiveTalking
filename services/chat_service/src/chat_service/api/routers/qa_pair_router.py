from __future__ import annotations

from chat_service.api.helpers.exception_handler import ExceptionHandler
from chat_service.application import QAPairService, QAPairInput
from chat_service.shared.utils import get_settings
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from authorization import CurrentToken

qa_pairs_router = APIRouter(prefix='/v1')
logger = get_logger(__name__)

settings = get_settings()

@qa_pairs_router.get('/qa_pairs')
async def get_qa_pairs_by_conversation_id(request: Request, current_token: CurrentToken, conversation_id: str, background_tasks: BackgroundTasks) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        qa_pair_service = QAPairService(
            request=request, settings=settings
        )

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await qa_pair_service.process(
            input=QAPairInput(
                conversation_id=conversation_id
            )
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'conversation_id': conversation_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )