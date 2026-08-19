"""
Custom exceptions for document loading.

WHY THESE EXIST:
Generic exceptions (ValueError, Exception) force calling code to guess
what went wrong from a string message. Custom exception TYPES let the
API layer catch each failure mode distinctly and return the right
HTTP status + a clear message to the frontend - e.g. "this file is
encrypted" (400, actionable) vs "this file is corrupted" (400,
different actionable message) vs an unexpected server bug (500).
"""


class DocumentLoadError(Exception):
    """Base class for all document loading failures."""


class InvalidFileTypeError(DocumentLoadError):
    """Raised when the uploaded file is not actually a valid PDF."""


class CorruptedFileError(DocumentLoadError):
    """Raised when the file claims to be a PDF but fails to parse."""


class EncryptedFileError(DocumentLoadError):
    """Raised when the PDF is password-protected and cannot be read."""


class FileTooLargeError(DocumentLoadError):
    """Raised when the uploaded file exceeds the configured size limit."""

class TooManyPagesError(DocumentLoadError):
    """Raised when the uploaded PDF exceeds the configured page limit."""


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingProviderConnectionError(EmbeddingError):
    """Raised when the embedding provider (e.g. Ollama) can't be reached."""


class EmbeddingProviderResponseError(EmbeddingError):
    """Raised when the provider responds, but with an unexpected/invalid shape."""


class VectorStoreError(Exception):
    """Base class for vector store failures."""

class LLMError(Exception):
    """Base class for LLM generation failures."""


class LLMProviderConnectionError(LLMError):
    """Raised when the LLM provider (e.g. Ollama) can't be reached."""


class LLMProviderResponseError(LLMError):
    """Raised when the provider responds, but with an unexpected/invalid shape."""
