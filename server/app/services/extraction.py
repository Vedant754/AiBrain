"""
Text extraction service.

RESPONSIBILITY:
Given a path to an already-validated PDF (Phase 3 already confirmed
it opens and isn't encrypted), extract clean, page-aware text:
  1. Pull block-level text per page (position-aware, per Phase 2's lab)
  2. Detect and strip repeated boilerplate (headers/footers)
  3. Rejoin hyphenated line-breaks
  4. Normalize excess whitespace

This service has ZERO knowledge of chunking, embeddings, or the API
layer - it only turns "a PDF" into "clean structured text."
"""

import re
from collections import Counter

import fitz

from app.models.schemas import ExtractedDocument, ExtractedPage

# A line appearing on at least this fraction of pages is treated as
# boilerplate (header/footer) rather than real content.
BOILERPLATE_FREQUENCY_THRESHOLD = 0.8

# Matches a word broken across a line by a hyphen, e.g. "transfor-\nmation"
HYPHEN_LINEBREAK_PATTERN = re.compile(r"(\w+)-\n(\w+)")

# Collapses runs of whitespace (but not single newlines we want to keep
# as paragraph-ish breaks) into single spaces.
MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")


def _extract_raw_pages(pdf: fitz.Document) -> list[str]:
    """
    Extracts raw text per page using block-level extraction, sorted into
    reading order (top-to-bottom, then left-to-right within a row band).
    This is the "smarter than naive word sorting" approach from Phase 2's
    lab, applied for real.
    """
    raw_pages = []
    for page in pdf:
        blocks = page.get_text("blocks")
        # Sort by (y0 rounded to nearest 5pt, x0) - grouping blocks into
        # rough horizontal bands before ordering left-to-right within
        # each band. This is what correctly separates column A's blocks
        # from column B's blocks instead of interleaving them.
        sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 5), b[0]))
        page_text = "\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())
        raw_pages.append(page_text)
    return raw_pages


def _detect_boilerplate_lines(raw_pages: list[str]) -> set[str]:
    """
    Finds lines that repeat across most pages - the signature of
    headers/footers rather than real content (see Step 1 diagram).
    """
    if len(raw_pages) < 2:
        # Boilerplate detection needs multiple pages to find repetition;
        # a single-page document has nothing to compare against.
        return set()

    line_counts: Counter[str] = Counter()
    for page_text in raw_pages:
        # Use a set per page so a line repeated twice on the SAME page
        # doesn't inflate its cross-page count.
        unique_lines_this_page = {ln.strip() for ln in page_text.split("\n") if ln.strip()}
        line_counts.update(unique_lines_this_page)

    threshold_count = max(2, int(len(raw_pages) * BOILERPLATE_FREQUENCY_THRESHOLD))
    return {line for line, count in line_counts.items() if count >= threshold_count}


def _clean_page_text(raw_text: str, boilerplate_lines: set[str]) -> str:
    """Strips detected boilerplate, rejoins hyphenated breaks, normalizes whitespace."""
    lines = [ln for ln in raw_text.split("\n") if ln.strip() not in boilerplate_lines]
    text = "\n".join(lines)

    text = HYPHEN_LINEBREAK_PATTERN.sub(r"\1\2", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
    return text.strip()


def extract_document(pdf_path: str, document_id: str) -> ExtractedDocument:
    """Full extraction pipeline for one document. See module docstring."""
    pdf = fitz.open(pdf_path)
    try:
        raw_pages = _extract_raw_pages(pdf)
    finally:
        pdf.close()

    boilerplate_lines = _detect_boilerplate_lines(raw_pages)

    pages = []
    for i, raw_text in enumerate(raw_pages):
        cleaned = _clean_page_text(raw_text, boilerplate_lines)
        pages.append(
            ExtractedPage(page_number=i + 1, text=cleaned, char_count=len(cleaned))
        )

    return ExtractedDocument(
        document_id=document_id,
        pages=pages,
        total_characters=sum(p.char_count for p in pages),
        stripped_boilerplate_lines=sorted(boilerplate_lines),
    )
