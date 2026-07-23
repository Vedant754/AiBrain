"""
Document loading service.

RESPONSIBILITY (and ONLY this):
Take raw uploaded bytes, validate they're a real, non-corrupted,
non-encrypted PDF within our size limit, persist them safely to disk
under a generated (not client-supplied) filename, and return metadata
about the document.

This service does NOT extract text content - see services/extraction.py
in Phase 4. Keeping this boundary strict is what makes each piece
independently testable.
"""

import os
import uuid

import fitz  # PyMuPDF

from app.core.config import settings
from app.core.exceptions import (
    CorruptedFileError,
    EncryptedFileError,
    FileTooLargeError,
    InvalidFileTypeError,
    TooManyPagesError,
)
from app.models.schemas import DocumentMetadata

# The real signature of a PDF file - the first bytes of any valid PDF.
# We check THIS, not the client-supplied Content-Type header, because
# the header is just a claim the client makes and cannot be trusted.
PDF_MAGIC_BYTES = b"%PDF-"


def _validate_pdf_bytes(file_bytes: bytes) -> None:
    """Raises InvalidFileTypeError if the bytes don't actually look like a PDF."""
    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        raise InvalidFileTypeError(
            "File does not appear to be a valid PDF (magic bytes mismatch)."
        )


def _validate_size(file_bytes: bytes) -> None:
    """Raises FileTooLargeError if the file exceeds our configured limit."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_upload_size_mb}MB upload limit."
        )


def _generate_safe_path(original_filename: str) -> tuple[str, str]:
    """
    Generates a collision-free, path-traversal-safe storage path.

    We deliberately DISCARD the client's filename for the actual stored
    path and generate our own UUID-based name. The original filename is
    kept only as metadata for display purposes, never used to construct
    a filesystem path.
    """
    os.makedirs(settings.upload_dir, exist_ok=True)
    document_id = str(uuid.uuid4())
    stored_path = os.path.join(settings.upload_dir, f"{document_id}.pdf")
    return document_id, stored_path


def load_pdf(file_bytes: bytes, original_filename: str) -> DocumentMetadata:
    """
    Validates, persists, and opens an uploaded PDF.

    Raises:
        InvalidFileTypeError: not actually a PDF
        FileTooLargeError: exceeds configured size limit
        CorruptedFileError: claims to be a PDF but fails to parse
        EncryptedFileError: valid PDF, but password-protected
    """
    _validate_size(file_bytes)
    _validate_pdf_bytes(file_bytes)

    document_id, stored_path = _generate_safe_path(original_filename)

    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    try:
        pdf = fitz.open(stored_path)
    except Exception as e:
        # PyMuPDF raises a generic exception on malformed files -
        # we translate it into OUR domain-specific exception type.
        raise CorruptedFileError(f"Failed to parse PDF: {e}") from e

    if pdf.is_encrypted:
        pdf.close()
        raise EncryptedFileError(
            "This PDF is password-protected and cannot be processed."
        )

    if pdf.page_count > settings.max_pages:
        pdf.close()
        raise TooManyPagesError(
            f"PDF has {pdf.page_count} pages, exceeding the limit of {settings.max_pages}."
        )

    metadata = DocumentMetadata(
        document_id=document_id,
        original_filename=original_filename,
        stored_path=stored_path,
        page_count=pdf.page_count,
        size_bytes=len(file_bytes),
        is_encrypted=False,
    )
    pdf.close()
    return metadata
