from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from app.api import deps
from app.evaluation.metrics import hit_at_k, mean_metric, recall_at_k
from app.models.schemas import EvalCase, EvalResult


def main(dataset_path: str = 'data/eval/golden_dataset.json', k: int = 5) -> None:
    cases = [EvalCase.model_validate(item) for item in json.loads(Path(dataset_path).read_text())]
    recalls, hits, latencies = [], [], []
    for case in cases:
        start = time.perf_counter()
        citations = deps.retriever().search(case.question, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved = [c.chunk_id for c in citations]
        recalls.append(recall_at_k(retrieved, case.gold_chunk_ids))
        hits.append(hit_at_k(retrieved, case.gold_chunk_ids))
    result = EvalResult(evaluated_at=datetime.utcnow(), cases=len(cases), recall_at_k=mean_metric(recalls), hit_at_k=mean_metric(hits), mean_latency_ms=round(mean_metric(latencies), 2))
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
