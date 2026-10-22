"""
Evaluation module for Information Retrieval metrics:
- MRR (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k
- Precision@k
- Latency (Mean, p50, p95)
"""

import math
from typing import Dict, List, Any
import numpy as np
from tabulate import tabulate


def compute_dcg(relevances: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def compute_ndcg_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k."""
    if not qrels or not any(v > 0 for v in qrels.values()):
        return 0.0

    retrieved_rels = [qrels.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    actual_dcg = compute_dcg(retrieved_rels, k)

    ideal_rels = sorted(qrels.values(), reverse=True)
    ideal_dcg = compute_dcg(ideal_rels, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_mrr(retrieved_ids: List[str], qrels: Dict[str, int]) -> float:
    """Computes Reciprocal Rank for the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Computes Recall at rank k."""
    relevant_ids = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant_ids:
        return 0.0
    retrieved_relevant = set(retrieved_ids[:k]) & relevant_ids
    return len(retrieved_relevant) / len(relevant_ids)


def compute_precision_at_k(retrieved_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Computes Precision at rank k."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for doc_id in top_k if qrels.get(doc_id, 0) > 0)
    return relevant_count / len(top_k)


class Evaluator:
    @staticmethod
    def evaluate_run(
        results: Dict[str, List[str]],
        qrels: Dict[str, Dict[str, int]],
        latencies_ms: List[float] = None,
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluates a complete retrieval run.
        results: {query_id: [retrieved_doc_id_1, retrieved_doc_id_2, ...]}
        qrels: {query_id: {doc_id: relevance_score}}
        """
        mrr_scores = []
        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for qid, retrieved in results.items():
            if qid not in qrels:
                continue
            query_qrels = qrels[qid]
            mrr_scores.append(compute_mrr(retrieved, query_qrels))
            ndcg_scores.append(compute_ndcg_at_k(retrieved, query_qrels, k=k))
            recall_scores.append(compute_recall_at_k(retrieved, query_qrels, k=k))
            precision_scores.append(compute_precision_at_k(retrieved, query_qrels, k=k))

        metrics = {
            f"NDCG@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            f"Recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
            f"Precision@{k}": float(np.mean(precision_scores)) if precision_scores else 0.0,
            "Evaluated Queries": len(mrr_scores),
        }

        if latencies_ms:
            metrics["Avg Latency (ms)"] = float(np.mean(latencies_ms))
            metrics["p50 Latency (ms)"] = float(np.percentile(latencies_ms, 50))
            metrics["p95 Latency (ms)"] = float(np.percentile(latencies_ms, 95))

        return metrics

    @staticmethod
    def print_comparison_table(metrics_by_system: Dict[str, Dict[str, float]], k: int = 10):
        """Prints a clean ASCII markdown comparison table across systems."""
        headers = [
            "Search Method",
            f"NDCG@{k}",
            "MRR",
            f"Recall@{k}",
            f"Precision@{k}",
            "Avg Latency (ms)",
            "p95 Latency (ms)",
        ]
        rows = []
        for system_name, metrics in metrics_by_system.items():
            rows.append([
                system_name,
                f"{metrics.get(f'NDCG@{k}', 0):.4f}",
                f"{metrics.get('MRR', 0):.4f}",
                f"{metrics.get(f'Recall@{k}', 0):.4f}",
                f"{metrics.get(f'Precision@{k}', 0):.4f}",
                f"{metrics.get('Avg Latency (ms)', 0):.2f}",
                f"{metrics.get('p95 Latency (ms)', 0):.2f}",
            ])
        print("\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n")
