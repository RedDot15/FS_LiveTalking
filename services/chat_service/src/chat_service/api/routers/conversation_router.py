from __future__ import annotations

from chat_service.api.helpers.exception_handler import ExceptionHandler
from chat_service.application import (
    ConversationService, 
    ConversationInput, 
    CreateConversationInput, 
    UpdateConversationInput,
    DeleteConversationInput
)
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
        body = ConversationInput(
            character_id=character_id,
            user_id=current_token.id
        )
        response = await conversation_service.process(
            input=body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': character_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@conversation_router.post('/conversations')
async def create_new_conversation(request: Request, current_token: CurrentToken, body: CreateConversationInput, background_tasks: BackgroundTasks) -> JSONResponse:

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
        body.user_id = current_token.id
        response = await conversation_service.create_new_conversation(
            input=body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': body.character_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )


@conversation_router.put('/conversations/{conversation_id}')
async def update_conversation(request: Request, current_token: CurrentToken, conversation_id: str, body: UpdateConversationInput, background_tasks: BackgroundTasks) -> JSONResponse:

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
        body.conversation_id = conversation_id
        body.user_id = current_token.id
        response = await conversation_service.update_conversation(
            input=body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'character_id': body.character_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )

@conversation_router.delete('/conversations/{conversation_id}')
async def delete_conversation(request: Request, current_token: CurrentToken, conversation_id: str, background_tasks: BackgroundTasks) -> JSONResponse:

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
        body = DeleteConversationInput()
        body.conversation_id = conversation_id
        body.user_id = current_token.id
        response = await conversation_service.delete_conversation(
            input=body
        )
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'user_id': current_token.id, 'conversation_id': body.conversation_id})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )