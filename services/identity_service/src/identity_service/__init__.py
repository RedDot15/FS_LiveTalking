from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute
from postgresql_client import PostgreSQL
from starlette.middleware.cors import CORSMiddleware

from identity_service.app.api.main import api_router
from identity_service.app.core.config import settings
import uvicorn


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.postgres = PostgreSQL(postgres_settings=settings.postgres_settings)

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

def main() -> None:
    uvicorn.run('identity_service:app', host='0.0.0.0', port=8762, reload=True)