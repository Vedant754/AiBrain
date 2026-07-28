"""
Chunking service.

RESPONSIBILITY:
Take an ExtractedDocument (page-tagged clean text from Phase 4) and
produce a list of Chunks suitable for embedding (Phase 6). Critically,
each chunk must retain which source page(s) it came from - this is
the "citability" requirement threaded through since Phase 4.

DESIGN NOTE:
RecursiveCharacterTextSplitter operates on ONE flat string. Our source
is naturally page-separated. So we:
  1. Concatenate all pages into one string, recording the character
     offset range each page occupies in that combined string.
  2. Run the splitter on the combined string.
  3. For each resulting chunk, re-locate its position in the combined
     string, then look up which page(s) that offset range falls into.
"""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.models.schemas import Chunk, ChunkingResult, ExtractedDocument

PAGE_JOIN_SEPARATOR = "\n\n"


def _build_full_text_with_page_map(
    document: ExtractedDocument,
) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Joins all page texts into one string and records, per page, the
    (start_offset, end_offset, page_number) range it occupies in that
    combined string.
    """
    full_text_parts = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0

    for page in document.pages:
        start = cursor
        full_text_parts.append(page.text)
        cursor += len(page.text)
        end = cursor
        page_ranges.append((start, end, page.page_number))

        full_text_parts.append(PAGE_JOIN_SEPARATOR)
        cursor += len(PAGE_JOIN_SEPARATOR)

    full_text = "".join(full_text_parts)
    return full_text, page_ranges


def _pages_for_offset_range(
    start: int, end: int, page_ranges: list[tuple[int, int, int]]
) -> tuple[int, int]:
    """
    Given a chunk's (start, end) character offsets in the combined text,
    finds which page(s) it overlaps. Returns (first_page, last_page) -
    identical values if the chunk doesn't cross a page boundary.
    """
    overlapping_pages = [
        page_number
        for (p_start, p_end, page_number) in page_ranges
        if start < p_end and end > p_start  # ranges overlap
    ]
    if not overlapping_pages:
        # Fallback: shouldn't happen if offsets were found correctly,
        # but fail safely rather than crash.
        return (page_ranges[0][2], page_ranges[0][2]) if page_ranges else (1, 1)
    return min(overlapping_pages), max(overlapping_pages)


def chunk_document(document: ExtractedDocument) -> ChunkingResult:
    full_text, page_ranges = _build_full_text_with_page_map(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    raw_chunks = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_from = 0  # where in full_text to resume searching for the next chunk

    for i, chunk_text in enumerate(raw_chunks):
        # Locate this chunk's actual position in the combined text.
        # We search starting slightly before the last known position
        # to correctly handle overlapping chunks (whose start is
        # BEFORE the previous chunk's end).
        found_at = full_text.find(chunk_text, max(0, search_from - settings.chunk_overlap))
        if found_at == -1:
            # Extremely defensive fallback - should not happen in practice
            # since the splitter only produces substrings of full_text.
            found_at = search_from

        start_offset = found_at
        end_offset = found_at + len(chunk_text)
        search_from = end_offset

        start_page, end_page = _pages_for_offset_range(start_offset, end_offset, page_ranges)

        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                chunk_index=i,
                text=chunk_text,
                char_count=len(chunk_text),
                start_page=start_page,
                end_page=end_page,
            )
        )

    return ChunkingResult(
        document_id=document.document_id,
        chunks=chunks,
        total_chunks=len(chunks),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
