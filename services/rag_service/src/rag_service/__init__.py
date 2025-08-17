from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from logger import get_logger
from logger import setup_logging
from chromadb_client import ChromaDB

from rag_service.shared.utils import get_settings
from rag_service.api.routers import rag_router
from rag_service.api.helpers import LoggingMiddleware

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.chromadb = ChromaDB(chromadb_setting=app.state.settings.chromadb)
    
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
    rag_router,
)

def main() -> None:
    uvicorn.run('rag_service:app', host='0.0.0.0', port=3005, reload=True)
