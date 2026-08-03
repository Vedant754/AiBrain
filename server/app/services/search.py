"""
Similarity search service.

RESPONSIBILITY:
Turn a user's natural-language query into ranked, page-attributed,
similarity-scored results - converting Chroma's raw distance numbers
into an honestly-labeled similarity, and applying an optional
threshold so weak matches can be discarded rather than force-fed to
the LLM later (Phase 10/11).

This is NOT the full RAG pipeline yet (that's Phase 9) - this module
only answers "what's in the vector store that's close to this query,"
nothing about prompt construction or generation.
"""

from app.db import vector_store
from app.models.schemas import SearchResult
from app.services import embedding


def similarity_search(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    similarity_threshold: float | None = None,
) -> list[SearchResult]:
    provider = embedding.get_embedding_provider()
    query_vector = embedding.embed_query(query, provider)

    where = {"document_id": document_id} if document_id else None
    raw = vector_store.query_collection(query_vector, top_k=top_k, where=where)

    # Chroma returns one list-of-lists per query embedding; we sent exactly
    # one query vector, so we only ever look at index [0].
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    results: list[SearchResult] = []
    for chunk_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=True
    ):
        # Chroma's cosine "distance" = 1 - cosine_similarity (see Step 1).
        # We convert back to similarity HERE, once, so nothing downstream
        # ever has to remember which direction "good" points in.
        similarity = 1 - distance

        if similarity_threshold is not None and similarity < similarity_threshold:
            continue

        results.append(
            SearchResult(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                chunk_index=metadata["chunk_index"],
                text=text,
                start_page=metadata["start_page"],
                end_page=metadata["end_page"],
                similarity=similarity,
            )
        )

    return results
