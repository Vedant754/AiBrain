"""
RAG pipeline orchestration service.

RESPONSIBILITY:
Coordinate retrieval -> prompt construction -> generation, in order,
and assemble the final answer + structured sources. This function
does no real work itself - it calls already-tested services in
sequence. See Phase 12 Step 1 for why this orchestration layer is
worth its own file rather than living in the route handler.

"""

import logging
import time

from app.core.config import settings
from app.models.schemas import AskResponse, RetrievedChunk, Source
from app.services import llm, prompt_builder, retrieval

logger = logging.getLogger(__name__)


def _build_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """
    Builds citations from OUR retrieval metadata - see Phase 12 Step 1.
    Groups all pages actually placed in context by document, deduplicated
    and sorted, regardless of what the model's own text says.
    """
    pages_by_doc: dict[str, set[int]] = {}
    for c in chunks:
        pages_by_doc.setdefault(c.document_id, set()).update(
            range(c.start_page, c.end_page + 1)
        )
    return [
        Source(document_id=doc_id, pages=sorted(pages))
        for doc_id, pages in pages_by_doc.items()
    ]



def answer_question(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    similarity_threshold: float | None = None,
) -> AskResponse:
    t0 = time.perf_counter()
    retrieval_result = retrieval.retrieve_context(
        query=query,
        top_k=top_k,
        document_id=document_id,
        similarity_threshold=similarity_threshold,
    )
    t1 = time.perf_counter()
    logger.info(
        "retrieval: %.3fs | has_context=%s | chunks=%d",
        t1 - t0,
        retrieval_result.has_relevant_context,
        len(retrieval_result.chunks),
    )

    bundle = prompt_builder.build_rag_prompt(query, retrieval_result)

    provider = llm.get_llm_provider()
    t2 = time.perf_counter()
    answer_text = provider.generate(bundle.system_prompt, bundle.user_prompt)
    t3 = time.perf_counter()
    logger.info("generation: %.3fs | model=%s", t3 - t2, provider.model_name)

    sources = (
        _build_sources(retrieval_result.chunks)
        if retrieval_result.has_relevant_context
        else []
    )

    return AskResponse(
        query=query,
        answer=answer_text,
        sources=sources,
        has_context=retrieval_result.has_relevant_context,
        model=provider.model_name,
        provider=settings.llm_provider,
        retrieval_time=t1 - t0,
        generation_time=t3 - t2,
    )
