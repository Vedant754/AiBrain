from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    app_name: str = "Personal AI Document Reader"
    version: str = "0.1.0"
    environment: str = "development"

    # --- LLM provider switch ---
    llm_provider: str = "ollama"

    # --- Ollama settings ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # --- OpenAI settings ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # --- Embedding model ---
    embedding_model: str = "nomic-embed-text"

    # --- Vector DB ---
    chroma_persist_dir: str = "./chroma_data"

    class Config:
        env_file = ".env"


# A single, importable instance used across the whole app.
settings = Settings()
