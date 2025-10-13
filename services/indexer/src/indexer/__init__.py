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
from chromadb_client import ChromaDB
from litellm import LiteLLMService

from indexer.api.routers import index_router
from indexer.shared.utils import get_settings
from indexer.api.helpers import LoggingMiddleware

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

glb_settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.minio_settings = glb_settings.minio
    app.state.minio_client = MinioConnection(setting = app.state.minio_settings)

    app.state.mongodb_settings = glb_settings.mongo
    app.state.mongodb_handler = MongoDBHandler(
        db=app.state.mongodb_settings.db,
        username=app.state.mongodb_settings.user,
        password=app.state.mongodb_settings.password,
        host=app.state.mongodb_settings.host,
        port=app.state.mongodb_settings.port,
    )

    app.state.chromadb_settings = glb_settings.chromadb
    app.state.chroma_client = ChromaDB(chromadb_setting = app.state.chromadb_settings)

    app.state.litellm_settings = glb_settings.litellm
    app.state.litellm_service = LiteLLMService(
        url=app.state.litellm_settings.url,
        model=app.state.litellm_settings.model,
        embedding_model=app.state.litellm_settings.embedding_model,
        frequency_penalty=app.state.litellm_settings.frequency_penalty,
        n=app.state.litellm_settings.n,
        presence_penalty=app.state.litellm_settings.presence_penalty,
        temperature=app.state.litellm_settings.temperature,
        top_p=app.state.litellm_settings.top_p,
        max_completion_tokens=app.state.litellm_settings.max_completion_tokens,
        encoding_format=app.state.litellm_settings.encoding_format,
        dimensions=app.state.litellm_settings.dimensions,
        max_length=app.state.litellm_settings.max_length,
    )
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
