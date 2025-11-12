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

from indexer.application import CharacterInputs, IndexerApplication, CharacterMongoDBInputs, ParserInput, IndexerApplicationInput

from fastapi import File, UploadFile, Form
from authorization import has_authority

logger = get_logger(__name__)
index_router = APIRouter(prefix="/v1")

@index_router.post('/indexing', dependencies=[Depends(has_authority(authority="INDEXING"))])
async def index(request: Request, 
                # character_inputs: CharacterInputs, 
                background_tasks: BackgroundTasks,
                body: IndexerApplicationInput,
                ) -> JSONResponse:
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    # Khoi tao
    logger.info("Bắt đầu luồng Indexer")
    try:
        index_application = IndexerApplication(
            request=request, 
        )
    except Exception as e:
        exception_handler.handle_exception("Lỗi khi khởi tạo Indexer",extra={e})
        raise e

    try:
        response = await index_application.process(
            inputs=body
        )
    except Exception as e:
        exception_handler.handle_exception("Lỗi khi process Indexer", extra={e})
        raise e
    
    return exception_handler.handle_success(
        # thanh cong
        jsonable_encoder(index_application)
    )