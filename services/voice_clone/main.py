from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.middleware.cors import CORSMiddleware

from api.routers import voice_clone_router
from shared.utils import get_settings
from infra.xtts.tts_funcs import TTSWrapper

from infra.minio_client import MinioConnection

from shared.logger import get_logger
logger = get_logger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Initialize TTS infrastructure
    app.state.settings = settings
    app.state.minio_client = MinioConnection(
        setting=settings.minio
    )
    app.state.tts_wrapper = TTSWrapper(
        output_folder=settings.output_folder,
        speaker_folder=settings.speaker_folder,
        model_folder=settings.model_folder,
        lowvram=settings.lowvram,
        model_version=settings.model_version,
        device=settings.device,
        deepspeed=settings.deepspeed,
        enable_cache_results=settings.enable_cache_results
    )
    
    # Load the model
    app.state.tts_wrapper.load_model()
    
    logger.info("Voice Clone service initialized successfully")
    yield
    logger.info("Voice Clone service shutting down")

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
    voice_clone_router,
)
