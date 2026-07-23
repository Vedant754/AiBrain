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

class DocumentMetadata(BaseModel):
    """
    Describes a successfully LOADED document. Notice this contains
    nothing about the document's actual text content yet - that's
    Phase 4's job. This is purely "the file exists, is valid, here's
    what we know about its shape."
    """

    document_id: str
    original_filename: str
    stored_path: str
    page_count: int
    size_bytes: int
    is_encrypted: bool


class UploadResponse(BaseModel):
    """What the API returns to the frontend after a successful upload."""

    message: str
    document: DocumentMetadata