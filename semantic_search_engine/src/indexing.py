"""
Indexing module for Qdrant.
Handles:
- Loading embedding models (Dense via BGE/MiniLM, Sparse via BM25)
- Collection creation with configurable HNSW parameters and similarity metrics
- Batch document embedding and uploading to Qdrant
"""

import os
from typing import Dict, List, Optional
from tqdm import tqdm

# Ensure fastembed caches models locally inside the project workspace
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastembed_cache")),
)

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"


class SearchIndexer:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "scifact_corpus",
        dense_model_name: str = DENSE_MODEL_NAME,
        sparse_model_name: str = SPARSE_MODEL_NAME,
    ):
        self.client = client
        self.collection_name = collection_name
        print(f"Loading dense embedding model: {dense_model_name}...")
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        print(f"Loading sparse embedding model: {sparse_model_name}...")
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    def setup_collection(
        self,
        vector_size: int = 384,
        distance: models.Distance = models.Distance.COSINE,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        recreate: bool = False,
    ):
        """Creates or recreates the Qdrant collection with dense and sparse vector configurations."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if recreate:
                print(f"Recreating collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Collection '{self.collection_name}' already exists. Skipping setup.")
                return

        print(
            f"Creating collection '{self.collection_name}' "
            f"(Metric: {distance}, HNSW m: {hnsw_m}, ef_construct: {hnsw_ef_construct})..."
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=distance,
                    hnsw_config=models.HnswConfigDiff(
                        m=hnsw_m,
                        ef_construct=hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"Collection '{self.collection_name}' created successfully.")

    def index_corpus(self, corpus: Dict[str, Dict], batch_size: int = 64):
        """Generates dense + sparse embeddings for the corpus and uploads in batches."""
        doc_ids = list(corpus.keys())
        total = len(doc_ids)
        print(f"Indexing {total} documents into '{self.collection_name}'...")

        for i in tqdm(range(0, total, batch_size), desc="Indexing batches"):
            batch_ids = doc_ids[i : i + batch_size]
            batch_texts = []
            batch_payloads = []

            for doc_id in batch_ids:
                doc = corpus[doc_id]
                title = doc.get("title", "")
                text = doc.get("text", "")
                combined_text = f"{title}\n{text}".strip()
                batch_texts.append(combined_text)
                batch_payloads.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": doc.get("metadata", {}),
                })

            # Generate dense embeddings (batch)
            dense_vectors = list(self.dense_model.embed(batch_texts))
            # Generate sparse embeddings (batch)
            sparse_vectors = list(self.sparse_model.embed(batch_texts))

            points = []
            for j, doc_id in enumerate(batch_ids):
                points.append(
                    models.PointStruct(
                        # Hash doc_id or use sequential integer ID for Qdrant point
                        id=i + j + 1,
                        vector={
                            "dense": dense_vectors[j].tolist(),
                            "sparse": models.SparseVector(
                                indices=sparse_vectors[j].indices.tolist(),
                                values=sparse_vectors[j].values.tolist(),
                            ),
                        },
                        payload=batch_payloads[j],
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        print(f"Successfully indexed {total} documents.")
