"""
Search module implementing:
1. Keyword / Lexical Search (BM25)
2. Dense Semantic Search (HNSW with tunable ef_search)
3. Hybrid Search via Reciprocal Rank Fusion (RRF) and score interpolation
"""

import os
import time
from typing import List, Dict, Any, Optional

# Ensure fastembed caches models locally inside the project workspace
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastembed_cache")),
)

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combines multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).
    RRF Score(d) = sum(1 / (k + rank(d)))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = item
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results = []
    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
        item = dict(doc_map[doc_id])
        item["score"] = score
        item["rank"] = rank
        fused_results.append(item)

    return fused_results


class HybridSearcher:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model: TextEmbedding,
        sparse_model: SparseTextEmbedding,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[models.Filter] = None,
    ) -> Dict[str, Any]:
        """Performs lexical search using Qdrant's sparse vector (BM25) index."""
        start_time = time.perf_counter()
        sparse_vector = list(self.sparse_model.embed([query]))[0]

        qdrant_sparse = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=qdrant_sparse,
            using="sparse",
            query_filter=filter_conditions,
            limit=top_k,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        hits = []
        for rank, p in enumerate(response.points, start=1):
            hits.append({
                "doc_id": p.payload.get("doc_id"),
                "title": p.payload.get("title"),
                "text": p.payload.get("text"),
                "score": p.score,
                "rank": rank,
            })

        return {"hits": hits, "latency_ms": latency_ms, "method": "keyword"}

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        ef_search: Optional[int] = None,
        filter_conditions: Optional[models.Filter] = None,
    ) -> Dict[str, Any]:
        """Performs dense semantic vector search with configurable HNSW search depth (ef)."""
        start_time = time.perf_counter()
        dense_vector = list(self.dense_model.embed([query]))[0].tolist()

        search_params = (
            models.SearchParams(hnsw_ef=ef_search) if ef_search is not None else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            search_params=search_params,
            query_filter=filter_conditions,
            limit=top_k,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        hits = []
        for rank, p in enumerate(response.points, start=1):
            hits.append({
                "doc_id": p.payload.get("doc_id"),
                "title": p.payload.get("title"),
                "text": p.payload.get("text"),
                "score": p.score,
                "rank": rank,
            })

        return {"hits": hits, "latency_ms": latency_ms, "method": "dense"}

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        fusion_k: int = 60,
        use_native_rrf: bool = True,
        filter_conditions: Optional[models.Filter] = None,
    ) -> Dict[str, Any]:
        """
        Performs hybrid search combining dense and sparse vectors via Reciprocal Rank Fusion.
        Supports both Qdrant server-side RRF and explicit client-side RRF.
        """
        start_time = time.perf_counter()

        dense_vector = list(self.dense_model.embed([query]))[0].tolist()
        sparse_vector = list(self.sparse_model.embed([query]))[0]
        qdrant_sparse = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        if use_native_rrf:
            # Native server-side fusion
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=qdrant_sparse,
                        using="sparse",
                        filter=filter_conditions,
                        limit=top_k * 3,
                    ),
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        filter=filter_conditions,
                        limit=top_k * 3,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            hits = []
            for rank, p in enumerate(response.points, start=1):
                hits.append({
                    "doc_id": p.payload.get("doc_id"),
                    "title": p.payload.get("title"),
                    "text": p.payload.get("text"),
                    "score": p.score,
                    "rank": rank,
                })
        else:
            # Client-side fusion
            kw_res = self.keyword_search(query, top_k=top_k * 2, filter_conditions=filter_conditions)
            dense_res = self.dense_search(query, top_k=top_k * 2, filter_conditions=filter_conditions)
            hits = reciprocal_rank_fusion([kw_res["hits"], dense_res["hits"]], k=fusion_k, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000

        return {"hits": hits, "latency_ms": latency_ms, "method": "hybrid"}
