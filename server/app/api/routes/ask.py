"""
The /ask endpoint - the full RAG pipeline, wired end to end.

This is what Phase 13's frontend will actually call. Every debug
endpoint from earlier phases (/search, /retrieve, /prompt/preview,
/generate) remains available for isolating a bad answer to a specific
stage - this endpoint doesn't replace them, it composes what they
each proved works.
"""

from fastapi import APIRouter, HTTPException

from app.core.exceptions import (
    EmbeddingProviderConnectionError,
    EmbeddingProviderResponseError,
    LLMProviderConnectionError,
    LLMProviderResponseError,
)
from app.models.schemas import AskResponse, SearchRequest
from app.services import rag_pipeline

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: SearchRequest) -> AskResponse:
    try:
        return rag_pipeline.answer_question(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            similarity_threshold=request.similarity_threshold,
        )
    except EmbeddingProviderConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except EmbeddingProviderResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except LLMProviderConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LLMProviderResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
