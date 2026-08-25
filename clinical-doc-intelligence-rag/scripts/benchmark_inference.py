from __future__ import annotations

import time

from app.core.config import get_settings
from app.llm.providers import build_provider
from app.models.schemas import EvidenceCitation


def main() -> None:
    settings = get_settings()
    provider = build_provider(settings)
    citation = EvidenceCitation(chunk_id='demo', document_id='demo_doc', pages=[1], quote='HbA1c result is 8.2%.')
    start = time.perf_counter()
    answer = provider.generate('What abnormal labs are present?', [citation], [])
    latency = (time.perf_counter() - start) * 1000
    print({'provider': settings.llm_provider, 'latency_ms': round(latency, 2), 'answer_chars': len(answer.answer), 'estimated_cost_usd': 0.0 if settings.llm_provider in {'mock', 'ollama'} else 'provider_free_tier'})


if __name__ == '__main__':
    main()
