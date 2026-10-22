# Semantic & Hybrid Search Engine with Qdrant

A modular Information Retrieval (IR) system that indexes a real corpus (`SciFact` from the BEIR benchmark) into **Qdrant**, implements **Hybrid Search** (Dense Semantic + Sparse BM25 via Reciprocal Rank Fusion), and compares search quality against Keyword and Dense baselines on a labeled query set.

---

## 🏛️ Architecture & Modules

```
semantic_search_engine/
├── data/                  # Downloaded BEIR corpus, queries, and qrels
├── src/
│   ├── dataset.py         # Module 1: BEIR dataset download, extraction, & parsing
│   ├── indexing.py        # Module 2: Embedding generation & Qdrant HNSW indexing
│   ├── search.py          # Module 3: Keyword, Dense, and Hybrid (RRF) search engines
│   └── evaluate.py        # Module 4: IR evaluation metrics (NDCG@k, MRR, Recall, Latency)
├── main.py                # Unified CLI interface
└── requirements.txt       # Dependencies
```

---

## ⚙️ How It Works (Concepts Implemented)

### 1. Dual-Vector Indexing (Dense + Sparse)
Each document point in Qdrant stores two vectors simultaneously:
- **`dense`**: Generated via `BAAI/bge-small-en-v1.5` (384 dimensions). Captures conceptual and semantic intent.
- **`sparse`**: Generated via BM25 term weighting (`Qdrant/bm25`). Captures exact keyword and lexical matches.

### 2. HNSW (Hierarchical Navigable Small World)
Configured with:
- `m = 16`: Number of bi-directional links per node in the graph layers.
- `ef_construct = 100`: Search depth during graph construction.
- `ef_search`: Dynamic search depth at query time. Controls the **Recall vs. Latency** trade-off.

### 3. Reciprocal Rank Fusion (RRF)
Merges ranked lists from lexical and dense retrievers:
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{lexical}, \text{dense}\}} \frac{1}{k + \text{rank}_m(d)}$$
Where $k=60$ prevents top-ranked outliers from dominating the final rank.

### 4. Metrics Evaluated
- **MRR (Mean Reciprocal Rank)**: Position of the first relevant document.
- **NDCG@10**: Discounted cumulative gain reflecting ranking order.
- **Recall@10**: Coverage of ground-truth relevant documents in top-10.
- **Latency**: Mean, p50, and p95 query response times.

---

## 🚀 Usage Guide

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Index the Corpus
Downloads SciFact (5,183 scientific abstracts) and indexes them into Qdrant:
```bash
# Using local embedded storage (zero external dependencies required)
python main.py index --storage-path ./qdrant_data

# Or against an existing Qdrant Docker instance:
python main.py index --qdrant-url http://localhost:6333
```

### 3. Run a Search Query
Compares Keyword, Dense, and Hybrid results side-by-side:
```bash
python main.py search "hypertension ACE inhibitors cardiovascular risk" --top-k 5
```

### 4. Run Benchmark & Evaluation
Evaluates all three methods on the test query set and prints a comparison table:
```bash
python main.py benchmark --eval-samples 500 --top-k 10
```
