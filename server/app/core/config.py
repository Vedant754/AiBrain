"""
Central configuration for the entire backend.

WHY THIS FILE EXISTS:
Every other module (services, routes, db clients) reads configuration
from THIS file only. No service should ever call `os.environ` directly.
This means when we later add OpenAI as an alternative to Ollama, we add
ONE field here, not go hunting through the codebase.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- General ---
    app_name: str = "Personal AI Document Reader"
    environment: str = "development"

    # --- LLM provider switch ---
    # This single field is what lets us swap Ollama -> OpenAI later
    # without touching any service logic (Phase 11 will use this).
    llm_provider: str = "ollama"  # "ollama" | "openai"

    # --- Ollama settings ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # --- OpenAI settings (used only if llm_provider == "openai") ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # --- Embedding model ---
    embedding_model: str = "nomic-embed-text"
    embedding_provider: str = "ollama"  # "ollama" | "openai"
    embedding_batch_size: int = 16
    embeddings_dir: str = "./data/embeddings"

     # --- Vector DB ---
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "documents"

    # --- File uploads ---
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 25
    max_pages: int = 500
    extracted_dir: str = "./data/extracted"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150
    chunks_dir: str = "./data/chunks"

    class Config:
        env_file = ".env"


# A single, importable instance used across the whole app.
settings = Settings()
