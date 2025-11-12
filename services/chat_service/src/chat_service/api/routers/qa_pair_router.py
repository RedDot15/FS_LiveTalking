from __future__ import annotations

from chat_service.api.helpers.exception_handler import ExceptionHandler
from chat_service.application import QAPairService, QAPairInput, CreateQAPairInput, UpdateQAPairInput
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
async def get_qa_pairs_by_conversation_id(request: Request, current_token: CurrentToken, body: QAPairInput, background_tasks: BackgroundTasks) -> JSONResponse:

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
            input=body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'conversation_id': body.conversation_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@qa_pairs_router.post('/qa_pairs')
async def create_new_qa_pair(request: Request, current_token: CurrentToken, body: CreateQAPairInput, background_tasks: BackgroundTasks) -> JSONResponse:

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
        body.user_id = current_token.id
        response = await qa_pair_service.create_new_qa_pair(
            input = body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': body.conversation_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@qa_pairs_router.put('/qa_pairs/{qa_pair_id}')
async def update_qa_pair(request: Request, current_token: CurrentToken, qa_pair_id: str, body: UpdateQAPairInput, background_tasks: BackgroundTasks) -> JSONResponse:

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
        body.qa_pair_id = qa_pair_id
        body.user_id = current_token.id
        response = await qa_pair_service.update_qa_pair(
            input = body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': body.qa_pair_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )