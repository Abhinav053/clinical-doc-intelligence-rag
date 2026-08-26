from __future__ import annotations

from statistics import mean


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    return 1.0 if set(retrieved_ids) & set(gold_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_ids)) / len(set(gold_ids))


def mean_metric(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0
