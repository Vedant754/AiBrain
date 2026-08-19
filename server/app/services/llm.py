"""
LLM generation service.

RESPONSIBILITY:
Send a system/user prompt pair to an LLM and return the generated
text. This module knows nothing about retrieval or prompt construction
- it only turns (system_prompt, user_prompt) into a string answer,
through whichever provider is configured.

NOTICE: this file's shape closely mirrors services/embedding.py from
Phase 6 (ABC + factory + provider implementations). That's the payoff
of establishing the pattern once - the second provider abstraction is
mostly copy-and-adapt, not a new design problem.
"""

from abc import ABC, abstractmethod

import httpx

from app.core.config import settings
from app.core.exceptions import LLMProviderConnectionError, LLMProviderResponseError


class LLMProvider(ABC):
    """Every LLM provider (Ollama, OpenAI, ...) implements this."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the model's generated answer as plain text."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class OllamaLLMProvider(LLMProvider):
    """Calls a local Ollama server's chat endpoint."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,  # non-streaming for now - see Phase 11 Step 1
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120.0,  # generation is slower than embedding - generous timeout
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise LLMProviderConnectionError(
                f"Could not reach Ollama at {self.base_url}. "
                f"Is Ollama running? Try `ollama serve` and `ollama pull {self.model}`."
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMProviderResponseError(
                f"Ollama returned an error: {e.response.status_code} {e.response.text}"
            ) from e

        data = response.json()
        message = data.get("message")
        if not message or "content" not in message:
            raise LLMProviderResponseError(
                f"Unexpected Ollama response shape: missing 'message.content'. Got: {data}"
            )
        return message["content"]


class OpenAILLMProvider(LLMProvider):
    """Stub mirroring OpenAIEmbeddingProvider from Phase 6 - same swap-readiness reasoning."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            "OpenAI LLM provider is scaffolded but not implemented yet. "
            "Set llm_provider='ollama' in settings for now."
        )


def get_llm_provider() -> LLMProvider:
    """Factory: returns whichever provider is configured - mirrors get_embedding_provider()."""
    if settings.llm_provider == "ollama":
        return OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    if settings.llm_provider == "openai":
        return OpenAILLMProvider(
            api_key=settings.openai_api_key or "", model=settings.openai_model
        )
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")
