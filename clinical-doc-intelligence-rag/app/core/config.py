from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    app_name: str = 'Clinical Document Intelligence RAG'
    api_prefix: str = '/api/v1'
    app_env: Literal['development', 'demo', 'production'] = 'demo'
    data_dir: Path = Path('data')
    upload_dir: Path = Path('data/uploads')
    artifacts_dir: Path = Path('data/artifacts')
    sqlite_path: Path = Path('data/clinical_rag.sqlite3')
    vector_dir: Path = Path('data/vector_index')
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 8
    rerank_k: int = 5
    dense_weight: float = 0.55
    bm25_weight: float = 0.45
    llm_provider: Literal['mock', 'groq', 'ollama'] = 'mock'
    groq_api_key: str = ''
    groq_model: str = 'llama-3.1-8b-instant'
    ollama_url: str = 'http://localhost:11434/api/generate'
    ollama_model: str = 'llama3.1:8b'
    enable_ocr: bool = True
    min_pdf_text_chars: int = 80
    allowed_origins: list[str] = ['http://localhost:8501', 'http://localhost:8000']


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for path in (settings.data_dir, settings.upload_dir, settings.artifacts_dir, settings.vector_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
