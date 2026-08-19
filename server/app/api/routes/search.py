"""
Search endpoint.

WHY THIS ROUTE IS SHAPED DIFFERENTLY FROM documents.py's ENDPOINTS:
Every endpoint so far has been /documents/{document_id}/something -
operating on ONE known document. Search is different: it can span
every document in the collection (or optionally be scoped to one via
the request body). That's why this lives in its own router rather
than being bolted onto documents.py.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import SearchRequest, SearchResponse, RetrievalResponse, PromptBundle,GenerationResult
from app.services import search, retrieval, prompt_builder
from app.core.exceptions import LLMProviderConnectionError, LLMProviderResponseError
from app.core.config import settings
from app.services import llm

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    results = search.similarity_search(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        similarity_threshold=request.similarity_threshold,
    )
    return SearchResponse(query=request.query, results=results, count=len(results))

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: SearchRequest) -> RetrievalResponse:
    """
    The full retrieval pipeline: search + neighbor expansion + reading
    order + character budget. This is what Phase 10 (prompt
    construction) will actually consume - /search stays available for
    debugging raw similarity results in isolation.
    """
    return retrieval.retrieve_context(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        similarity_threshold=request.similarity_threshold,
    )

@router.post("/prompt/preview", response_model=PromptBundle)
async def preview_prompt(request: SearchRequest) -> PromptBundle:
    """
    DEBUG/DEVELOPMENT ENDPOINT: builds and returns the exact prompt
    that would be sent to an LLM, WITHOUT calling any LLM (Phase 11
    wires up actual generation). Being able to inspect the literal
    prompt text - not just trust it's "probably fine" - is essential
    for debugging retrieval-to-prompt issues before blaming the model.
    """
    retrieval_result = retrieval.retrieve_context(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        similarity_threshold=request.similarity_threshold,
    )
    return prompt_builder.build_rag_prompt(request.query, retrieval_result)

@router.post("/generate", response_model=GenerationResult)
async def generate_from_prompt(bundle: PromptBundle) -> GenerationResult:
    """
    DEBUG/DEVELOPMENT ENDPOINT (Phase 11 scope): takes an already-built
    PromptBundle (e.g. from /prompt/preview) and calls the configured
    LLM provider. Deliberately standalone, NOT chained to retrieval
    here - Phase 12 wires search -> retrieve -> prompt -> generate
    into one single end-to-end endpoint.
    """
    provider = llm.get_llm_provider()
    try:
        text = provider.generate(bundle.system_prompt, bundle.user_prompt)
    except LLMProviderConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LLMProviderResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return GenerationResult(
        text=text, model=provider.model_name, provider=settings.llm_provider
    )
