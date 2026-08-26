from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, metrics, query
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version='0.1.0')
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
    for router in (health.router, documents.router, query.router, metrics.router):
        app.include_router(router, prefix=settings.api_prefix)
    return app
