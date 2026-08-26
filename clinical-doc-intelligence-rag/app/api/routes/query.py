import time

from fastapi import APIRouter

from app.api import deps
from app.models.schemas import QueryRequest, QueryResponse
from app.safety.phi_redactor import redact_phi
from app.safety.query_classifier import classify_query

router = APIRouter(tags=['query'])


@router.post('/query', response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    redacted_question, phi_flags = redact_phi(request.question)
    safety_flags = sorted(set(phi_flags + classify_query(redacted_question)))
    citations = deps.retriever().search(redacted_question, k=request.top_k)
    answer = deps.llm_provider().generate(redacted_question, citations, safety_flags) if request.use_llm else deps.llm_provider().generate(redacted_question, citations, safety_flags)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return QueryResponse(question=redacted_question, answer=answer, latency_ms=latency_ms, retrieved_chunks=len(citations))
