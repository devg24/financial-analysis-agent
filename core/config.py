from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """OpenAI-compatible LLM endpoint (e.g. Ollama at localhost:11434/v1)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    openai_model: str = "llama3.1"
    openai_temperature: float = 0.0

    # Earnings-call pipeline
    fmp_api_key: str = ""
    earnings_chroma_path: str = "./chroma_db"
