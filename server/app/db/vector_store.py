"""
Vector store service - our only point of contact with ChromaDB.

RESPONSIBILITY:
Persist embedded chunks (vector + metadata) into a durable, indexed
collection, and provide a minimal query capability. Deep querying
mechanics (top_k tuning, distance interpretation, filter syntax) are
Phase 8's focus - this phase only proves storage + basic retrieval work.

DESIGN NOTES:
- ONE shared collection for the whole app; document_id lives in
  metadata so we can filter to one document OR search across all of
  them later (Phase 12/13) without restructuring storage.
- We explicitly configure "hnsw:space": "cosine" - Chroma's default is
  squared L2, which would silently contradict the cosine-similarity
  mental model we built in Phase 6 if left unconfigured.
- We never let Chroma compute embeddings itself - we always pass
  vectors we already produced via services/embedding.py, keeping that
  module the single source of truth for "how text becomes a vector."
"""

import chromadb

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.models.schemas import EmbeddingResult

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """
    Returns a singleton PersistentClient. Singleton because ChromaDB's
    PersistentClient opens files on disk - repeatedly re-instantiating
    it per-request would be wasteful and can cause file-lock contention.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection():
    """
    Returns our single shared collection, creating it on first use with
    cosine distance explicitly configured (see module docstring).
    """
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_embedding_result(result: EmbeddingResult) -> int:
    """
    Stores (or updates, if re-run) all embedded chunks from one
    document's EmbeddingResult into the shared collection.

    "Upsert" (not plain insert) matters here: if a document is
    re-processed (e.g. after improving chunking), re-running this
    should overwrite the old vectors for the same chunk_ids, not
    duplicate them.
    """
    if not result.embedded_chunks:
        return 0

    collection = get_collection()

    try:
        collection.upsert(
            ids=[c.chunk_id for c in result.embedded_chunks],
            embeddings=[c.embedding for c in result.embedded_chunks],
            documents=[c.text for c in result.embedded_chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "start_page": c.start_page,
                    "end_page": c.end_page,
                }
                for c in result.embedded_chunks
            ],
        )
    except Exception as e:
        raise VectorStoreError(f"Failed to upsert into vector store: {e}") from e

    return len(result.embedded_chunks)


def smoke_test_query(query_embedding: list[float], top_k: int = 3) -> dict:
    """
    Minimal query used ONLY to prove storage/indexing works this phase.
    Real query design (top_k tuning, filters, score interpretation)
    is Phase 8's subject - don't build retrieval logic on this yet.
    """
    collection = get_collection()
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
