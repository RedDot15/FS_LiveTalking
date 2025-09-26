from __future__ import annotations

import io
import json

from fastapi import APIRouter
from fastapi import Request
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from indexer.api.helpers.exception_handler import ExceptionHandler
from indexer.api.helpers.exception_handler import ResponseMessage
from logger import get_logger

from indexer.application import CharacterInputs, IndexerApplication, CharacterMongoDBInputs, ParserInput

from fastapi import File, UploadFile, Form

logger = get_logger(__name__)
index_router = APIRouter(prefix="/v1")

@index_router.post('/indexing')
async def index(request: Request, 
                # character_inputs: CharacterInputs, 
                background_tasks: BackgroundTasks,
                name: str = Form(...),
                knowledge_file: UploadFile = File(...),
                avatar_image: UploadFile = File(...),
                ) -> JSONResponse:
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    # Khoi tao
    try:
        index_application = IndexerApplication(
            request=request, 
        )
    except Exception as e:
        raise e

    return exception_handler.handle_success(
        # thanh cong
        jsonable_encoder(index_application)
    )