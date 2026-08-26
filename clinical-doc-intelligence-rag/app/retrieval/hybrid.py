from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.models.schemas import Chunk, EvidenceCitation


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.replace('/', ' ').replace('-', ' ').split() if t.strip()]


class HybridRetriever:
    def __init__(self, model_name: str, vector_dir: Path, dense_weight: float = 0.55, bm25_weight: float = 0.45) -> None:
        self.model = SentenceTransformer(model_name)
        self.vector_dir = vector_dir
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            return
        embeddings = self.model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False).astype('float32')
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        faiss.write_index(self.index, str(self.vector_dir / 'dense.faiss'))
        with open(self.vector_dir / 'chunks.pkl', 'wb') as f:
            pickle.dump(chunks, f)

    def load(self) -> bool:
        index_path = self.vector_dir / 'dense.faiss'
        chunk_path = self.vector_dir / 'chunks.pkl'
        if not index_path.exists() or not chunk_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(chunk_path, 'rb') as f:
            self.chunks = pickle.load(f)
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])
        return True

    def search(self, query: str, k: int = 5) -> list[EvidenceCitation]:
        if self.index is None or not self.chunks:
            self.load()
        if self.index is None or not self.chunks:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype('float32')
        dense_scores, dense_idx = self.index.search(query_vec, min(max(k * 4, k), len(self.chunks)))
        dense_map = {int(idx): float(score) for idx, score in zip(dense_idx[0], dense_scores[0], strict=False) if idx >= 0}
        bm25_scores = self.bm25.get_scores(_tokenize(query)) if self.bm25 else np.zeros(len(self.chunks))
        max_bm25 = max(float(np.max(bm25_scores)), 1.0)
        combined: list[tuple[int, float]] = []
        candidate_ids = set(dense_map) | set(np.argsort(bm25_scores)[-k * 4:].tolist())
        for idx in candidate_ids:
            score = self.dense_weight * dense_map.get(idx, 0.0) + self.bm25_weight * (float(bm25_scores[idx]) / max_bm25)
            combined.append((idx, score))
        combined.sort(key=lambda item: item[1], reverse=True)
        citations = []
        for idx, score in combined[:k]:
            chunk = self.chunks[idx]
            quote = chunk.text[:350]
            citations.append(EvidenceCitation(chunk_id=chunk.chunk_id, document_id=chunk.document_id, pages=chunk.page_numbers, quote=quote, score=round(score, 4)))
        return citations
