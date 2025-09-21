from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from api.routers import sadtalker_router
from shared.utils import get_settings
from infra.minio_client import MinioConnection
from infra.minio_client import MinioSettings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.minio_client = MinioConnection(
        setting=settings.minio
    )
    
    yield

app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(CorrelationIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(
    sadtalker_router,
)
