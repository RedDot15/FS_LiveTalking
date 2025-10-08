from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from logger import get_logger
from logger import setup_logging

from chat_service.shared.utils import get_settings
from chat_service.api.routers import chat_router, conversation_router, qa_pairs_router
from chat_service.api.helpers import LoggingMiddleware

from llm_client import LLMService
from mongo_client import MongoDBHandler

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.llm = LLMService(settings=app.state.settings.llm)
    app.state.mongodb_client = MongoDBHandler(
        db=app.state.settings.mongodb.db,
        username=app.state.settings.mongodb.username,
        password=app.state.settings.mongodb.password,
        host=app.state.settings.mongodb.host,
        port=app.state.settings.mongodb.port,
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

app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(qa_pairs_router)

def main() -> None:
    uvicorn.run('chat_service:app', host='0.0.0.0', port=3000, reload=True)
