from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from logger import get_logger
from logger import setup_logging

from chat_service.shared.utils import get_settings
from chat_service.api.routers import conversation_router, qa_pairs_router, chat_router
from chat_service.api.helpers import LoggingMiddleware

from llm_client import LLMService
from mongo_client import MongoDBHandler
from litellm import LiteLLMService

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.llm = LLMService(settings=app.state.settings.llm)
    app.state.mongodb_client = MongoDBHandler(
        db=app.state.settings.mongo.db,
        username=app.state.settings.mongo.user,
        password=app.state.settings.mongo.password,
        host=app.state.settings.mongo.host,
        port=app.state.settings.mongo.port,
    )
    app.state.litellm_settings = settings.litellm
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
    allow_origins=['http://localhost:8080'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(qa_pairs_router)
app.include_router(chat_router)

def main() -> None:
    uvicorn.run('chat_service:app', host='0.0.0.0', port=3000, reload=True)
