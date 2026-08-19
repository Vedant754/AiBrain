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

import os

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.exceptions import (
    CorruptedFileError,
    EncryptedFileError,
    FileTooLargeError,
    InvalidFileTypeError,
    TooManyPagesError,
    EmbeddingError,
    EmbeddingProviderConnectionError,
    EmbeddingProviderResponseError,
    VectorStoreError,

)
from app.models.schemas import IngestionResult, UploadResponse
from app.models.schemas import ExtractedDocument
from app.models.schemas import ChunkingResult
from app.models.schemas import EmbeddingResult, EmbeddedChunk
from app.models.schemas import StoreResult,IngestionResult
from app.db import vector_store
from app.services.document_loader import load_pdf
from app.core.config import settings
from app.services import document_loader, extraction, chunking, embedding, ingestion_pipeline

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

@router.post("/documents/process", response_model=IngestionResult)
async def process_document(file: UploadFile) -> IngestionResult:
    """
    The full ingestion pipeline in one call: load -> extract -> chunk ->
    embed -> store. This is what the frontend (Phase 13) actually calls.
    The individual /upload, /extract, /chunk, /embed, /store endpoints
    remain available for isolating a problem to a specific stage -
    exactly the same relationship /api/ask has to /search, /retrieve,
    /prompt/preview, /generate (Phase 12).
    """
    file_bytes = await file.read()

    try:
        result = ingestion_pipeline.process_document(
            file_bytes, original_filename=file.filename or "unnamed.pdf"
        )
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InvalidFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except CorruptedFileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EncryptedFileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EmbeddingProviderConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except EmbeddingProviderResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return result

@router.post("/documents/{document_id}/extract", response_model=ExtractedDocument)
async def extract_document_text(document_id: str) -> ExtractedDocument:
    """
    Extracts clean, page-aware text from a previously uploaded document.

    Split from upload deliberately (per Phase 4's load-vs-extract
    distinction) - a client could re-trigger extraction (e.g. after we
    improve the cleaning logic) without re-uploading the file.
    """
    try:
        pdf_path = document_loader.get_pdf_path(document_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    result = extraction.extract_document(pdf_path, document_id=document_id)

    # Persist the extraction result so Phase 5 (chunking) can load it
    # without re-running extraction every time.
    os.makedirs(settings.extracted_dir, exist_ok=True)
    out_path = os.path.join(settings.extracted_dir, f"{document_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    return result

@router.post("/documents/{document_id}/chunk", response_model=ChunkingResult)
async def chunk_document_text(document_id: str) -> ChunkingResult:
    """
    Chunks a previously extracted document.

    Split from extraction (same reasoning as extraction being split
    from upload): chunk_size/overlap are tunable, and we want to be
    able to re-chunk without re-extracting from the PDF.
    """
    extracted_path = os.path.join(settings.extracted_dir, f"{document_id}.json")
    if not os.path.exists(extracted_path):
        raise HTTPException(
            status_code=404,
            detail=f"No extraction found for document_id {document_id}. Run /extract first.",
        )

    with open(extracted_path, encoding="utf-8") as f:
        extracted_document = ExtractedDocument.model_validate_json(f.read())

    result = chunking.chunk_document(extracted_document)

    os.makedirs(settings.chunks_dir, exist_ok=True)
    out_path = os.path.join(settings.chunks_dir, f"{document_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    return result

@router.post("/documents/{document_id}/embed", response_model=EmbeddingResult)
async def embed_document_chunks(document_id: str) -> EmbeddingResult:
    """Embeds a previously chunked document's text and persists the vectors."""
    chunks_path = os.path.join(settings.chunks_dir, f"{document_id}.json")
    if not os.path.exists(chunks_path):
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for document_id {document_id}. Run /chunk first.",
        )

    with open(chunks_path, encoding="utf-8") as f:
        chunking_result = ChunkingResult.model_validate_json(f.read())

    provider = embedding.get_embedding_provider()
    texts = [c.text for c in chunking_result.chunks]

    try:
        vectors = embedding.embed_document_chunks(texts, provider)
    except EmbeddingProviderConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except EmbeddingProviderResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    embedded_chunks = [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            text=c.text,
            start_page=c.start_page,
            end_page=c.end_page,
            embedding=vec,
            embedding_model=provider.model_name,
        )
        for c, vec in zip(chunking_result.chunks, vectors, strict=True)
    ]

    result = EmbeddingResult(
        document_id=document_id,
        embedded_chunks=embedded_chunks,
        embedding_dimension=len(vectors[0]) if vectors else 0,
        embedding_model=provider.model_name,
        provider=settings.embedding_provider,
    )

    os.makedirs(settings.embeddings_dir, exist_ok=True)
    out_path = os.path.join(settings.embeddings_dir, f"{document_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    return result

@router.post("/documents/{document_id}/store", response_model=StoreResult)
async def store_document_embeddings(document_id: str) -> StoreResult:
    """
    Loads a previously computed EmbeddingResult and upserts it into the
    shared ChromaDB collection. This is the final step turning a raw
    PDF into something actually searchable.
    """
    embeddings_path = os.path.join(settings.embeddings_dir, f"{document_id}.json")
    if not os.path.exists(embeddings_path):
        raise HTTPException(
            status_code=404,
            detail=f"No embeddings found for document_id {document_id}. Run /embed first.",
        )

    with open(embeddings_path, encoding="utf-8") as f:
        embedding_result = EmbeddingResult.model_validate_json(f.read())

    try:
        count = vector_store.upsert_embedding_result(embedding_result)
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return StoreResult(
        document_id=document_id,
        chunks_stored=count,
        collection_name=settings.chroma_collection_name,
    )

