"""
Embedding service.

RESPONSIBILITY:
Convert chunk text into vectors, via a swappable PROVIDER. This module
knows nothing about chunking or vector storage - it only turns
List[str] into List[List[float]], through whichever provider is
configured.

WHY AN ABSTRACT BASE CLASS:
Your requirements explicitly asked for easy Ollama -> OpenAI swapping.
The way we guarantee that "easy" stays true over time is by making
EVERY provider satisfy the exact same interface. Calling code (the
route, in this phase) never imports OllamaEmbeddingProvider directly -
it asks a factory function for "the configured provider" and only
talks to it through the shared interface.
"""

from abc import ABC, abstractmethod

import httpx

from app.core.config import settings
from app.core.exceptions import (
    EmbeddingProviderConnectionError,
    EmbeddingProviderResponseError,
)

# nomic-embed-text is an ASYMMETRIC model - see Phase 6 Step 1. Text
# being indexed and text being searched need different prefixes for
# the model to produce well-aligned vectors between the two.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingProvider(ABC):
    """Every embedding provider (Ollama, OpenAI, ...) implements this."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, in the same order."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Calls a local Ollama server's embedding endpoint."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise EmbeddingProviderConnectionError(
                f"Could not reach Ollama at {self.base_url}. "
                f"Is Ollama running? Try `ollama serve` and `ollama pull {self.model}`."
            ) from e
        except httpx.HTTPStatusError as e:
            raise EmbeddingProviderResponseError(
                f"Ollama returned an error: {e.response.status_code} {e.response.text}"
            ) from e

        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise EmbeddingProviderResponseError(
                f"Unexpected Ollama response shape: missing 'embeddings' key. Got: {data}"
            )
        return embeddings


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Stub for the OpenAI swap your requirements asked for. Not wired up
    or tested yet (no API key configured in this project) - but because
    it satisfies the same EmbeddingProvider interface, activating it
    later is a one-line change in get_embedding_provider(), nothing
    else in the app needs to know.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "OpenAI embedding provider is scaffolded but not implemented yet. "
            "Set embedding_provider='ollama' in settings for now."
        )


def get_embedding_provider() -> EmbeddingProvider:
    """Factory: returns whichever provider is configured, so callers never hardcode one."""
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url, model=settings.embedding_model
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key or "", model=settings.embedding_model
        )
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")


def embed_document_chunks(
    texts: list[str], provider: EmbeddingProvider
) -> list[list[float]]:
    """
    Embeds chunk texts for INDEXING (not querying - see QUERY_PREFIX,
    used later in Phase 9 for the user's question instead).
    Batches requests per settings.embedding_batch_size to amortize
    network round-trip cost rather than one HTTP call per chunk.
    """
    all_embeddings: list[list[float]] = []
    batch_size = settings.embedding_batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prefixed_batch = [DOCUMENT_PREFIX + t for t in batch]
        batch_embeddings = provider.embed_batch(prefixed_batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
