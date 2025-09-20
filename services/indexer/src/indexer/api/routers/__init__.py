from __future__ import annotations

import io
import json

from fastapi import APIRouter, BackgroundTasks, Request
from indexer.api.helpers.exception_handler import ExceptionHandler, ResponseMessage
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
    
    
