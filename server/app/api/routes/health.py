"""
Health check endpoint.

WHY THIS EXISTS:
Before we ever touch PDFs or embeddings, we need proof that:
1. FastAPI boots correctly
2. Our config layer loads correctly
3. The frontend can successfully reach the backend across ports

This is the "hello world" of a production API - nearly every real
backend has a /health endpoint used by monitoring, load balancers,
and (for us) a sanity check during development.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.models.schemas import VersionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns basic app status. No business logic - just proves the wiring works."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
    )

@router.get("/version", response_model=VersionResponse)
async def version_check() -> VersionResponse:
    """Returns the app version."""
    return VersionResponse(
        app_name=settings.app_name,
        version=settings.version
    )
