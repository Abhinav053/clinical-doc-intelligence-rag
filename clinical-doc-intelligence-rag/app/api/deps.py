from functools import lru_cache

from app.core.config import get_settings
from app.document_ai.chunker import ProvenanceChunker
from app.document_ai.parser import ClinicalDocumentParser
from app.llm.providers import build_provider
from app.retrieval.hybrid import HybridRetriever
from app.storage.sqlite_store import SQLiteStore


@lru_cache
def store():
    return SQLiteStore(get_settings().sqlite_path)


@lru_cache
def parser():
    return ClinicalDocumentParser(get_settings())


@lru_cache
def chunker():
    s = get_settings()
    return ProvenanceChunker(s.chunk_size, s.chunk_overlap)


@lru_cache
def retriever():
    s = get_settings()
    r = HybridRetriever(s.embedding_model, s.vector_dir, s.dense_weight, s.bm25_weight)
    r.load()
    return r


@lru_cache
def llm_provider():
    return build_provider(get_settings())
