"""
Search endpoint.

WHY THIS ROUTE IS SHAPED DIFFERENTLY FROM documents.py's ENDPOINTS:
Every endpoint so far has been /documents/{document_id}/something -
operating on ONE known document. Search is different: it can span
every document in the collection (or optionally be scoped to one via
the request body). That's why this lives in its own router rather
than being bolted onto documents.py.
"""

from fastapi import APIRouter

from app.models.schemas import SearchRequest, SearchResponse
from app.services import search

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
