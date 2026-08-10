"""
Pydantic schemas define the SHAPE of data flowing in and out of our API.

WHY THIS MATTERS:
FastAPI uses these to auto-validate incoming requests (reject bad data
before it ever reaches our business logic) and to auto-generate API
documentation. Think of these as contracts: "this endpoint returns
exactly this shape, guaranteed."

This file will grow significantly in later phases (UploadResponse,
QueryRequest, QueryResponse, etc.). For Phase 1 we only need a health
check contract to prove the whole stack is wired correctly.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str

class VersionResponse(BaseModel):
    app_name: str
    version: str

class DocumentMetadata(BaseModel):
    """
    Describes a successfully LOADED document. Notice this contains
    nothing about the document's actual text content yet - that's
    Phase 4's job. This is purely "the file exists, is valid, here's
    what we know about its shape."
    """

    document_id: str
    original_filename: str
    stored_path: str
    page_count: int
    size_bytes: int
    is_encrypted: bool


class UploadResponse(BaseModel):
    """What the API returns to the frontend after a successful upload."""

    message: str
    document: DocumentMetadata

class ExtractedPage(BaseModel):
    """Clean text extracted from a single page, with its page number preserved."""

    page_number: int  # 1-indexed - matches how humans refer to pages
    text: str
    char_count: int


class ExtractedDocument(BaseModel):
    """
    Full extraction result for a document. This is the object every
    later phase (chunking, embedding, retrieval) builds on top of -
    never the raw PDF again.
    """

    document_id: str
    pages: list[ExtractedPage]
    total_characters: int
    stripped_boilerplate_lines: list[str]

class Chunk(BaseModel):
    """
    One retrievable unit of text. This is what eventually gets embedded
    (Phase 6) and stored in the vector DB (Phase 7). Page attribution
    is preserved so retrieved chunks remain citable back to a source
    page, even though chunk boundaries don't align with page boundaries.
    """

    chunk_id: str
    document_id: str
    chunk_index: int  # 0-indexed position among this document's chunks
    text: str
    char_count: int
    start_page: int  # first page this chunk's text overlaps
    end_page: int  # last page this chunk's text overlaps (== start_page if it doesn't span pages)


class ChunkingResult(BaseModel):
    document_id: str
    chunks: list[Chunk]
    total_chunks: int
    chunk_size: int
    chunk_overlap: int

class EmbeddedChunk(BaseModel):
    """A Chunk plus its embedding vector - the unit that goes into the vector DB (Phase 7)."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int
    embedding: list[float]
    embedding_model: str


class EmbeddingResult(BaseModel):
    document_id: str
    embedded_chunks: list[EmbeddedChunk]
    embedding_dimension: int
    embedding_model: str
    provider: str

class StoreResult(BaseModel):
    document_id: str
    chunks_stored: int
    collection_name: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: str | None = None  # if set, restrict search to one document
    similarity_threshold: float | None = None  # e.g. 0.3 - discard weaker matches


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int
    similarity: float  # 1.0 = identical direction, -1.0 = opposite; NOT a probability


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    count: int

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int
    similarity: float | None  # None for chunks pulled in purely via neighbor expansion
    is_neighbor: bool  # True if included for context, not because it matched the query


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    has_relevant_context: bool
    total_characters: int
    truncated: bool  # True if the character budget cut off additional relevant chunks

class PromptBundle(BaseModel):
    """
    The final, ready-to-send prompt - Phase 11 sends this to an LLM
    verbatim, no further assembly needed. Separated into system/user
    because that's how essentially every modern chat-completion API
    (Ollama, OpenAI) expects messages structured.
    """

    system_prompt: str
    user_prompt: str
    has_context: bool  # mirrors RetrievalResponse.has_relevant_context, for transparency