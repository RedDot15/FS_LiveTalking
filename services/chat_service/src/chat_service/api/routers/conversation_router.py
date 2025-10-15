from __future__ import annotations

from chat_service.api.helpers.exception_handler import ExceptionHandler
from chat_service.application import ConversationService, ConversationInput, CreateConversationInput
from chat_service.shared.utils import get_settings
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from logger import get_logger
from authorization import CurrentToken

conversation_router = APIRouter(prefix='/v1')
logger = get_logger(__name__)

settings = get_settings()

@conversation_router.get('/conversations')
async def get_conversations_by_character_id(request: Request, current_token: CurrentToken, character_id: str, background_tasks: BackgroundTasks) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        conversation_service = ConversationService(
            request=request, settings=settings
        )

    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await conversation_service.process(
            input=ConversationInput(
                user_id=current_token.id,
                character_id=character_id
            )
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': character_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@conversation_router.post('/conversations')
async def create_new_conversation(request: Request, current_token: CurrentToken, question: str, character_id: str, sessionid: int, background_tasks: BackgroundTasks) -> JSONResponse:

    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )

    try:
        conversation_service = ConversationService(
            request=request, settings=settings
        )
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during application initialization: {str(e)}',
            extra={},
        )

    try:
        response = await conversation_service.create_new_conversation(
            input=CreateConversationInput(
                question=question,
                user_id=current_token.id,
                character_id=character_id,
                sessionid=sessionid
            )
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': character_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )