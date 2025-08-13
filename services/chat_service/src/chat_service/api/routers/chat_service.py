from __future__ import annotations

from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from chat_service.application import ChatServiceInput
from chat_service.application import ChatServiceApplication
from chat_service.api.helpers.exception_handler import ExceptionHandler
from chat_service.shared.utils import get_settings

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
            input=ChatServiceInput(
                question=chatbot_input.question,
                character_name=chatbot_input.character_name
            )
        )
        
    except Exception as e:
        return exception_handler.handle_exception(e=str(e), extra={'question': chatbot_input.question})

    return exception_handler.handle_success(
        jsonable_encoder(
            response,
        )
    )