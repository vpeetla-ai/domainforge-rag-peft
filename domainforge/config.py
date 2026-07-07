from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    domainforge_api_key: str = ""
    data_root: Path = Path("data")
    corpus_dir: Path = Path("data/corpus/sop_documents")
    manifest_path: Path = Path("data/manifests/sop_intent_map.json")
    mock_llm: bool = True
    retriever_mode: str = "memory"  # memory | chroma | hybrid
    chroma_path: Path = Path("data/indexes/chroma_s1")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_adapter_model: str = "domainforge-triage"
    adapter_registry_path: Path = Path("adapters/registry.json")
    cors_origins: str = "http://localhost:3000,https://domainforge-rag-peft.vercel.app"


def get_settings() -> Settings:
    return Settings()
