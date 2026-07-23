"""
Pydantic schemas define the SHAPE of data flowing in and out of our API.

WHY THIS MATTERS:
FastAPI uses these to auto-validate incoming requests (reject bad data
before it ever reaches our business logic) and to auto-generate API
documentation. Think of these as contracts: "this endpoint returns
exactly this shape, guaranteed."

This file will grow significantly in later phases (UploadResponse,
QueryRequest, QueryResponse, etc.). For Phase 1 we only need a health
check contract to prove the whole stack is wired correctly.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str

class VersionResponse(BaseModel):
    app_name: str
    version: str
