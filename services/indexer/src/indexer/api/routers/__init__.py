from __future__ import annotations

import io
import json

from fastapi import APIRouter
from fastapi import Request
from fastapi import BackgroundTasks

from indexer.api.helpers.exception_handler import ExceptionHandler
from indexer.api.helpers.exception_handler import ResponseMessage
from logger import get_logger


logger = get_logger(__name__)
index_router = APIRouter()

@index_router.post('/index')
def index_knowledge(request: Request,
                    background_tasks: BackgroundTasks):
    
    exception_handler = ExceptionHandler(
        logger=logger.bind(),
        service_name=__name__,
    )
    
    
