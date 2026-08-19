"""
Ingestion pipeline orchestration service.

RESPONSIBILITY:
Coordinate load -> extract -> chunk -> embed -> store, in order, calling
the SAME underlying service functions the individual per-stage routes
already use (services/document_loader.py, extraction.py, chunking.py,
embedding.py, db/vector_store.py). This is not new logic - it's the
exact same orchestration-layer pattern as rag_pipeline.py (Phase 12),
applied to ingestion.

We deliberately still persist each stage's intermediate JSON to disk,
exactly like the individual routes do - so /extract, /chunk, /embed
etc. remain fully usable afterward for debugging a specific document,
even if it was originally processed through this single call.
"""

import logging
import os
import time

from app.core.config import settings
from app.db import vector_store
from app.models.schemas import IngestionResult
from app.services import chunking, document_loader, embedding, extraction
from app.models.schemas import EmbeddedChunk, EmbeddingResult

logger = logging.getLogger(__name__)


def process_document(file_bytes: bytes, original_filename: str) -> IngestionResult:
    t0 = time.perf_counter()

    metadata = document_loader.load_pdf(file_bytes, original_filename)
    document_id = metadata.document_id

    extracted = extraction.extract_document(metadata.stored_path, document_id=document_id)
    os.makedirs(settings.extracted_dir, exist_ok=True)
    with open(
        os.path.join(settings.extracted_dir, f"{document_id}.json"), "w", encoding="utf-8"
    ) as f:
        f.write(extracted.model_dump_json(indent=2))

    chunked = chunking.chunk_document(extracted)
    os.makedirs(settings.chunks_dir, exist_ok=True)
    with open(
        os.path.join(settings.chunks_dir, f"{document_id}.json"), "w", encoding="utf-8"
    ) as f:
        f.write(chunked.model_dump_json(indent=2))

    provider = embedding.get_embedding_provider()
    texts = [c.text for c in chunked.chunks]
    vectors = embedding.embed_document_chunks(texts, provider)


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
        for c, vec in zip(chunked.chunks, vectors, strict=True)
    ]
    embedding_result = EmbeddingResult(
        document_id=document_id,
        embedded_chunks=embedded_chunks,
        embedding_dimension=len(vectors[0]) if vectors else 0,
        embedding_model=provider.model_name,
        provider=settings.embedding_provider,
    )
    os.makedirs(settings.embeddings_dir, exist_ok=True)
    with open(
        os.path.join(settings.embeddings_dir, f"{document_id}.json"), "w", encoding="utf-8"
    ) as f:
        f.write(embedding_result.model_dump_json(indent=2))

    chunks_stored = vector_store.upsert_embedding_result(embedding_result)

    t1 = time.perf_counter()
    logger.info(
        "ingestion pipeline: %.3fs | document_id=%s | pages=%d | chunks=%d",
        t1 - t0,
        document_id,
        metadata.page_count,
        chunked.total_chunks,
    )

    return IngestionResult(
        document_id=document_id,
        original_filename=metadata.original_filename,
        page_count=metadata.page_count,
        total_chunks=chunked.total_chunks,
        chunks_stored=chunks_stored,
        embedding_model=provider.model_name,
        message="Document processed and ready for search.",
    )
