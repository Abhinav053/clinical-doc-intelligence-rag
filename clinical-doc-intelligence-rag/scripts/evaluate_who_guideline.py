from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    docs = len(corpus_tokens)
    df: Counter[str] = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {term: math.log((docs + 1) / (freq + 0.5)) + 1 for term, freq in df.items()}


def score(query_tokens: list[str], doc_tokens: list[str], idf_map: dict[str, float]) -> float:
    counts = Counter(doc_tokens)
    doc_len = max(len(doc_tokens), 1)
    total = 0.0
    for term in query_tokens:
        tf = counts[term]
        if tf:
            total += idf_map.get(term, 1.0) * (tf / math.sqrt(doc_len))
    return total


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def evaluate(dataset_path: Path, k_values: tuple[int, ...] = (1, 3, 5)) -> dict:
    dataset = json.loads(dataset_path.read_text())
    chunks = dataset["chunks"]
    questions = dataset["questions"]
    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    idf_map = idf(corpus_tokens)
    details = []
    latency_ms = []

    aggregate = {
        f"hit_at_{k}": [] for k in k_values
    } | {
        f"recall_at_{k}": [] for k in k_values
    } | {
        f"precision_at_{k}": [] for k in k_values
    } | {
        f"ndcg_at_{k}": [] for k in k_values
    }
    mrr_values = []

    for item in questions:
        start = time.perf_counter()
        q_tokens = tokenize(item["question"])
        ranked = sorted(
            (
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": score(q_tokens, tokens, idf_map),
                    "pages": chunk["pages"],
                }
                for chunk, tokens in zip(chunks, corpus_tokens, strict=True)
            ),
            key=lambda row: row["score"],
            reverse=True,
        )
        latency_ms.append((time.perf_counter() - start) * 1000)
        gold = set(item["gold_chunk_ids"])
        ranked_ids = [row["chunk_id"] for row in ranked]
        first_rank = next((idx + 1 for idx, chunk_id in enumerate(ranked_ids) if chunk_id in gold), None)
        mrr_values.append(0.0 if first_rank is None else 1.0 / first_rank)

        for k in k_values:
            top_ids = ranked_ids[:k]
            hits = len(set(top_ids) & gold)
            aggregate[f"hit_at_{k}"].append(1.0 if hits else 0.0)
            aggregate[f"recall_at_{k}"].append(hits / len(gold))
            aggregate[f"precision_at_{k}"].append(hits / k)
            relevances = [1 if chunk_id in gold else 0 for chunk_id in top_ids]
            ideal = [1] * min(len(gold), k) + [0] * max(k - len(gold), 0)
            aggregate[f"ndcg_at_{k}"].append(dcg(relevances) / max(dcg(ideal), 1e-9))

        details.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_chunk_ids": item["gold_chunk_ids"],
                "top_3": ranked[:3],
                "first_relevant_rank": first_rank,
            }
        )

    metrics = {key: round(mean(values), 4) for key, values in aggregate.items()}
    metrics["mrr"] = round(mean(mrr_values), 4)
    metrics["mean_latency_ms"] = round(mean(latency_ms), 4)
    metrics["questions"] = len(questions)
    metrics["chunks"] = len(chunks)
    metrics["document_id"] = dataset["document"]["document_id"]
    metrics["document_title"] = dataset["document"]["title"]
    return {"metrics": metrics, "details": details}


if __name__ == "__main__":
    result = evaluate(Path("data/eval/who_pneumonia_eval.json"))
    output_path = Path("data/eval/who_pneumonia_eval_results.json")
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote detailed results to {output_path}")
