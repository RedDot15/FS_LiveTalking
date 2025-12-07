from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from logger import get_logger
from logger import setup_logging

from mongo_client import MongoDBHandler
from minio_client import MinioConnection

from character_service.shared.utils import get_settings
from character_service.api.routers import character_management_router
from character_service.api.helpers import LoggingMiddleware

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.mongodb_client = MongoDBHandler(
        db=app.state.settings.mongodb.db,
        username=app.state.settings.mongodb.user,
        password=app.state.settings.mongodb.password,
        host=app.state.settings.mongodb.host,
        port=app.state.settings.mongodb.port,
    )
    app.state.minio_client = MinioConnection(setting = app.state.settings.minio)
    
    yield

app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(CorrelationIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(
    character_management_router,
)

def main() -> None:
    uvicorn.run('character_service:app', host='0.0.0.0', port=3006, reload=True)
