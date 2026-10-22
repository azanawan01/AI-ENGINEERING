"""
Main entry point for Semantic Search Engine.
Provides commands for:
1. Indexing corpus into Qdrant (Dense + Sparse HNSW)
2. Interactive / Single Query Search
3. Benchmark & Comparison (Keyword vs Dense vs Hybrid on labeled query set)
"""

import argparse
import os
import sys

# Ensure fastembed caches models locally inside the project workspace
os.environ["FASTEMBED_CACHE_PATH"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "fastembed_cache")
)

from qdrant_client import QdrantClient, models

from src.dataset import download_and_extract, load_corpus, load_queries, load_qrels
from src.indexing import SearchIndexer
from src.search import HybridSearcher
from src.evaluate import Evaluator


def get_qdrant_client(args) -> QdrantClient:
    """Initializes Qdrant client based on CLI args (in-memory, local directory, or remote server)."""
    if args.qdrant_url:
        print(f"Connecting to Qdrant at {args.qdrant_url}...")
        return QdrantClient(url=args.qdrant_url)
    elif args.storage_path:
        os.makedirs(args.storage_path, exist_ok=True)
        print(f"Connecting to local disk Qdrant at {args.storage_path}...")
        return QdrantClient(path=args.storage_path)
    else:
        print("Using in-memory Qdrant instance...")
        return QdrantClient(":memory:")


def run_index(args):
    """Step 1: Download dataset and index into Qdrant."""
    dataset_path = download_and_extract(dataset_name=args.dataset, data_dir=args.data_dir)
    corpus = load_corpus(dataset_path, max_docs=args.max_docs)
    print(f"Loaded {len(corpus)} documents from corpus.")

    client = get_qdrant_client(args)
    indexer = SearchIndexer(
        client=client,
        collection_name=args.collection,
        dense_model_name=args.dense_model,
        sparse_model_name=args.sparse_model,
    )

    indexer.setup_collection(
        vector_size=args.vector_size,
        distance=models.Distance.COSINE,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construct=args.hnsw_ef_construct,
        recreate=args.recreate,
    )

    indexer.index_corpus(corpus, batch_size=args.batch_size)
    print("\nIndexing complete!")


def run_search(args):
    """Step 2: Run a single query across search modes."""
    client = get_qdrant_client(args)
    from fastembed import TextEmbedding, SparseTextEmbedding

    dense_model = TextEmbedding(model_name=args.dense_model)
    sparse_model = SparseTextEmbedding(model_name=args.sparse_model)
    searcher = HybridSearcher(client, args.collection, dense_model, sparse_model)

    query = args.query
    print(f"\nQuery: '{query}'\n" + "=" * 60)

    for method in ["keyword", "dense", "hybrid"]:
        if method == "keyword":
            res = searcher.keyword_search(query, top_k=args.top_k)
        elif method == "dense":
            res = searcher.dense_search(query, top_k=args.top_k, ef_search=args.ef_search)
        else:
            res = searcher.hybrid_search(query, top_k=args.top_k)

        print(f"\n--- [{method.upper()} SEARCH] (Latency: {res['latency_ms']:.2f} ms) ---")
        for hit in res["hits"]:
            title = hit.get("title") or "No title"
            preview = hit.get("text", "")[:120].replace("\n", " ")
            print(f"#{hit['rank']} [Doc {hit['doc_id']}] (Score: {hit['score']:.4f}) | {title} | {preview}...")


def run_benchmark(args):
    """Step 3: Run benchmark evaluation comparing Keyword vs Dense vs Hybrid on labeled query set."""
    dataset_path = download_and_extract(dataset_name=args.dataset, data_dir=args.data_dir)
    queries = load_queries(dataset_path)
    qrels = load_qrels(dataset_path, split="test")

    client = get_qdrant_client(args)

    # Get all indexed doc_ids in the collection to ensure fair evaluation
    print(f"Inspecting indexed documents in '{args.collection}'...")
    scroll_res, _ = client.scroll(
        collection_name=args.collection,
        limit=10000,
        with_payload=["doc_id"],
        with_vectors=False,
    )
    indexed_doc_ids = {p.payload["doc_id"] for p in scroll_res if p.payload and "doc_id" in p.payload}
    print(f"Found {len(indexed_doc_ids)} unique indexed documents in Qdrant.")

    # Filter qrels to only consider relevant docs that are actually indexed
    adjusted_qrels = {}
    for qid, docs in qrels.items():
        relevant_in_index = {did: score for did, score in docs.items() if did in indexed_doc_ids}
        if relevant_in_index:
            adjusted_qrels[qid] = relevant_in_index

    valid_qids = [qid for qid in queries if qid in adjusted_qrels]
    if args.eval_samples and args.eval_samples < len(valid_qids):
        valid_qids = valid_qids[:args.eval_samples]

    print(f"Selected {len(valid_qids)} labeled queries with ground-truth docs present in index.")
    from fastembed import TextEmbedding, SparseTextEmbedding

    dense_model = TextEmbedding(model_name=args.dense_model)
    sparse_model = SparseTextEmbedding(model_name=args.sparse_model)
    searcher = HybridSearcher(client, args.collection, dense_model, sparse_model)

    results_by_system = {"Keyword (BM25)": {}, "Dense Semantic": {}, "Hybrid (RRF)": {}}
    latencies_by_system = {"Keyword (BM25)": [], "Dense Semantic": [], "Hybrid (RRF)": []}

    print(f"Running evaluation over {len(valid_qids)} test queries...")
    for qid in valid_qids:
        query_text = queries[qid]

        # 1. Keyword Search
        kw_res = searcher.keyword_search(query_text, top_k=args.top_k)
        results_by_system["Keyword (BM25)"][qid] = [h["doc_id"] for h in kw_res["hits"]]
        latencies_by_system["Keyword (BM25)"].append(kw_res["latency_ms"])

        # 2. Dense Semantic Search
        dense_res = searcher.dense_search(query_text, top_k=args.top_k, ef_search=args.ef_search)
        results_by_system["Dense Semantic"][qid] = [h["doc_id"] for h in dense_res["hits"]]
        latencies_by_system["Dense Semantic"].append(dense_res["latency_ms"])

        # 3. Hybrid Search
        hybrid_res = searcher.hybrid_search(query_text, top_k=args.top_k)
        results_by_system["Hybrid (RRF)"][qid] = [h["doc_id"] for h in hybrid_res["hits"]]
        latencies_by_system["Hybrid (RRF)"].append(hybrid_res["latency_ms"])

    # Compute IR metrics
    metrics_summary = {}
    for system_name, run_results in results_by_system.items():
        metrics = Evaluator.evaluate_run(
            results=run_results,
            qrels=adjusted_qrels,
            latencies_ms=latencies_by_system[system_name],
            k=args.top_k,
        )
        metrics_summary[system_name] = metrics

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS (Top-{args.top_k} Retrieval on {args.dataset.upper()})")
    print("=" * 70)
    Evaluator.print_comparison_table(metrics_summary, k=args.top_k)


def main():
    parser = argparse.ArgumentParser(description="Semantic & Hybrid Search Engine with Qdrant")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Common parent parser
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--collection", type=str, default="scifact_corpus")
    common.add_argument("--qdrant-url", type=str, default=None, help="e.g. http://localhost:6333")
    common.add_argument("--storage-path", type=str, default="./qdrant_data")
    common.add_argument("--dense-model", type=str, default="BAAI/bge-small-en-v1.5")
    common.add_argument("--sparse-model", type=str, default="Qdrant/bm25")

    # Index command
    idx_parser = subparsers.add_parser("index", parents=[common], help="Index corpus into Qdrant")
    idx_parser.add_argument("--dataset", type=str, default="scifact")
    idx_parser.add_argument("--data-dir", type=str, default="data")
    idx_parser.add_argument("--max-docs", type=int, default=None, help="Limit documents to index (optional)")
    idx_parser.add_argument("--batch-size", type=int, default=64)
    idx_parser.add_argument("--vector-size", type=int, default=384)
    idx_parser.add_argument("--hnsw-m", type=int, default=16)
    idx_parser.add_argument("--hnsw-ef-construct", type=int, default=100)
    idx_parser.add_argument("--recreate", action="store_true", help="Recreate collection if exists")

    # Search command
    search_parser = subparsers.add_parser("search", parents=[common], help="Run a single search query")
    search_parser.add_argument("query", type=str, help="Search query text")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--ef-search", type=int, default=64)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", parents=[common], help="Run evaluation on labeled queries")
    bench_parser.add_argument("--dataset", type=str, default="scifact")
    bench_parser.add_argument("--data-dir", type=str, default="data")
    bench_parser.add_argument("--eval-samples", type=int, default=50, help="Number of queries to evaluate (e.g. 50 or all)")
    bench_parser.add_argument("--top-k", type=int, default=10)
    bench_parser.add_argument("--ef-search", type=int, default=64)

    args = parser.parse_args()

    if args.mode == "index":
        run_index(args)
    elif args.mode == "search":
        run_search(args)
    elif args.mode == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()
