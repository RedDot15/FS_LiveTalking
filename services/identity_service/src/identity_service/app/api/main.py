from fastapi import APIRouter

from identity_service.app.api.routes import login, users, roles

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(roles.router)

