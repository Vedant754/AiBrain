"""
Document upload endpoint.

WHY THIS FILE IS THIN:
Notice this route does almost nothing itself - it reads bytes from
the request, calls the service, and translates exceptions into HTTP
responses. ALL real logic (validation, saving, parsing) lives in
document_loader.py. This is the "routers stay thin" rule from Phase 1,
now actually paying off: we can unit-test load_pdf() with zero HTTP
machinery involved at all.
"""

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.exceptions import (
    CorruptedFileError,
    EncryptedFileError,
    FileTooLargeError,
    InvalidFileTypeError,
    TooManyPagesError,
)
from app.models.schemas import UploadResponse
from app.services.document_loader import load_pdf

router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    file_bytes = await file.read()

    try:
        metadata = load_pdf(file_bytes, original_filename=file.filename or "unnamed.pdf")
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InvalidFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except CorruptedFileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EncryptedFileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return UploadResponse(
        message="Document uploaded and validated successfully.",
        document=metadata,
    )
