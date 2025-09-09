from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from logger import get_logger
from logger import setup_logging
from minio_client import MinioConnection
from mongo_client import MongoDBHandler

from indexer.api.routers import index_router
from indexer.shared.utils import get_minio_settings, get_mongodb_settings
from indexer.api.helpers import LoggingMiddleware

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

minio_settings = get_minio_settings()
mongodb_settings = get_mongodb_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.minio_settings = minio_settings
    app.state.minio_client = MinioConnection(setting = app.state.minio_settings.minio)
    app.state.mongodb_settings = mongodb_settings
    app.state.mongodb_handler = MongoDBHandler(mongo_settings = app.state.mongodb_settings.mongo)
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
    index_router,
)

def main() -> None:
    uvicorn.run('indexer:app', host='0.0.0.0', port=3036, reload=True)
