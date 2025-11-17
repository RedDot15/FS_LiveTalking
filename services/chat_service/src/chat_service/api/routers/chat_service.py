from __future__ import annotations

from chat_service.api.helpers.exception_handler import ExceptionHandler
from chat_service.application import ChatServiceApplication, ChatServiceInput
from chat_service.shared.utils import get_settings
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger

chat_router = APIRouter(prefix='/v1')
logger = get_logger(__name__)

settings = get_settings()

@chat_router.post('/chatbot')
async def query(request: Request, chatbot_input: ChatServiceInput, background_tasks: BackgroundTasks) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        chatbot_service = ChatServiceApplication(
            request=request, settings=settings
        )

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await chatbot_service.process(
            input=chatbot_input
        )

    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'question': chatbot_input.question})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )