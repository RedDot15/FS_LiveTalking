from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from realistic import LipReal
from realistic import BaseReal
from realistic import load_model
from realistic import load_avatar

from typing import Dict

from logger import get_logger
from logger import setup_logging
from fastapi.staticfiles import StaticFiles

from livetalking.api.helpers import LoggingMiddleware
from livetalking.shared.utils import get_settings

from livetalking.api.routers import livetalking_router

settings = get_settings()

setup_logging(json_logs=False, log_level='INFO')
logger = get_logger('api')

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    app.state.settings = settings
    app.state.nerfreals = {}
    app.state.model = load_model(path=settings.model)
    app.state.avatar = load_avatar(avatar_id=settings.avatar_id)
    app.state.pcs = set()
    
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
    livetalking_router,    
)

app.mount("/", StaticFiles(directory="web", html=True), name="web")

def main() -> None:
    uvicorn.run('livetalking:app', host='0.0.0.0', port=8010, reload=True)
