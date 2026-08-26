from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceBlock(BaseModel):
    document_id: str
    page_number: int
    block_id: str
    block_type: Literal['pdf_text', 'ocr_text', 'table']
    text: str
    bbox: list[float] | None = None
    confidence: float | None = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    page_numbers: list[int]
    text: str
    source_blocks: list[str]
    source_type: str


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    blocks: int
    chunks: int
    tables: int
    indexed: bool


class EvidenceCitation(BaseModel):
    chunk_id: str
    document_id: str
    pages: list[int]
    quote: str
    score: float | None = None


class ClinicalAnswer(BaseModel):
    answer: str
    key_findings: list[str] = Field(default_factory=list)
    citations: list[EvidenceCitation] = Field(default_factory=list)
    confidence: Literal['low', 'moderate', 'high'] = 'low'
    limitations: str
    safety_flags: list[str] = Field(default_factory=list)
    disclaimer: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: ClinicalAnswer
    latency_ms: float
    retrieved_chunks: int


class EvalCase(BaseModel):
    id: str
    question: str
    expected_answer: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_document_ids: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    evaluated_at: datetime
    cases: int
    recall_at_k: float
    hit_at_k: float
    mean_latency_ms: float
