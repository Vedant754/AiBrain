"""
Retrieval service.

RESPONSIBILITY:
Take Phase 8's raw similarity_search results and turn them into the
FINAL set of chunks that will actually go into the LLM's prompt
(Phase 10): expand with neighboring chunks for continuity, reorder
into document reading order, enforce a character budget, and
explicitly signal when nothing relevant was found at all.
"""

from app.core.config import settings
from app.db import vector_store
from app.models.schemas import RetrievalResponse, RetrievedChunk, SearchResult
from app.services.search import similarity_search


def _expand_with_neighbors(results: list[SearchResult]) -> list[RetrievedChunk]:
    """
    For each search hit, also pull chunk_index-1 and +1 from the SAME
    document (if they exist and aren't already present), to restore
    context that chunking may have severed (Phase 5's core tradeoff).
    """
    chunks_by_id: dict[str, RetrievedChunk] = {
        r.chunk_id: RetrievedChunk(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            text=r.text,
            start_page=r.start_page,
            end_page=r.end_page,
            similarity=r.similarity,
            is_neighbor=False,
        )
        for r in results
    }

    if not settings.expand_neighbors:
        return list(chunks_by_id.values())

    # Group requested neighbor indices per document, so we do ONE
    # lookup per document rather than one per chunk (fewer round trips
    # to Chroma - same batching philosophy as Phase 6's embedding calls).
    neighbor_indices_by_doc: dict[str, set[int]] = {}
    for r in results:
        wanted = {r.chunk_index - 1, r.chunk_index + 1}
        neighbor_indices_by_doc.setdefault(r.document_id, set()).update(wanted)

    for document_id, indices in neighbor_indices_by_doc.items():
        # Don't re-fetch chunks we already have from the search hits themselves.
        already_have = {
            c.chunk_index for c in chunks_by_id.values() if c.document_id == document_id
        }
        to_fetch = [i for i in indices if i >= 0 and i not in already_have]
        if not to_fetch:
            continue

        raw = vector_store.get_chunks_by_index(document_id, to_fetch)
        for chunk_id, text, metadata in zip(
            raw["ids"], raw["documents"], raw["metadatas"], strict=True
        ):
            if chunk_id in chunks_by_id:
                continue
            chunks_by_id[chunk_id] = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                chunk_index=metadata["chunk_index"],
                text=text,
                start_page=metadata["start_page"],
                end_page=metadata["end_page"],
                similarity=None,
                is_neighbor=True,
            )

    return list(chunks_by_id.values())


def _order_for_reading(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Groups by document, and within each document sorts by chunk_index -
    i.e. original reading order, NOT similarity rank. A stitched
    neighbor-expanded passage should read top-to-bottom coherently.
    Documents themselves are ordered by their best (max) similarity
    among their own chunks, so the most relevant document's chunks
    appear first.
    """
    doc_best_similarity: dict[str, float] = {}
    for c in chunks:
        if c.similarity is not None:
            doc_best_similarity[c.document_id] = max(
                doc_best_similarity.get(c.document_id, -1.0), c.similarity
            )

    return sorted(
        chunks,
        key=lambda c: (-doc_best_similarity.get(c.document_id, -1.0), c.document_id, c.chunk_index),
    )


def _apply_character_budget(
    chunks: list[RetrievedChunk], max_chars: int
) -> tuple[list[RetrievedChunk], bool]:
    """Keeps adding chunks (in the given order) until the budget would be exceeded."""
    kept: list[RetrievedChunk] = []
    total = 0
    truncated = False

    for c in chunks:
        if total + len(c.text) > max_chars:
            truncated = True
            break
        kept.append(c)
        total += len(c.text)

    return kept, truncated


def retrieve_context(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    similarity_threshold: float | None = None,
) -> RetrievalResponse:
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.default_similarity_threshold
    )

    results = similarity_search(
        query=query,
        top_k=top_k,
        document_id=document_id,
        similarity_threshold=threshold,
    )

    if not results:
        return RetrievalResponse(
            query=query,
            chunks=[],
            has_relevant_context=False,
            total_characters=0,
            truncated=False,
        )

    expanded = _expand_with_neighbors(results)
    ordered = _order_for_reading(expanded)
    final_chunks, truncated = _apply_character_budget(ordered, settings.max_context_chars)

    return RetrievalResponse(
        query=query,
        chunks=final_chunks,
        has_relevant_context=True,
        total_characters=sum(len(c.text) for c in final_chunks),
        truncated=truncated,
    )
