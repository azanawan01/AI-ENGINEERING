"""
Dataset loader and downloader for evaluation benchmark.
Supports downloading and extracting BEIR datasets (e.g., SciFact)
and parsing corpus, queries, and qrels (relevance judgments).
"""

import os
import zipfile
import json
import csv
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm

BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
}

def download_and_extract(dataset_name: str = "scifact", data_dir: str = "data") -> str:
    """Downloads and unzips a BEIR benchmark dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, dataset_name)
    if os.path.exists(target_dir):
        print(f"Dataset '{dataset_name}' already exists at {target_dir}.")
        return target_dir

    url = BEIR_URLS.get(dataset_name, f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip")
    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")

    print(f"Downloading {dataset_name} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        desc=f"Downloading {dataset_name}.zip",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Dataset ready at {target_dir}")
    return target_dir


def load_corpus(dataset_path: str, max_docs: int = None) -> Dict[str, Dict]:
    """Loads corpus.jsonl -> {doc_id: {'title': str, 'text': str}}"""
    corpus_file = os.path.join(dataset_path, "corpus.jsonl")
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            corpus[doc_id] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }
    return corpus


def load_queries(dataset_path: str) -> Dict[str, str]:
    """Loads queries.jsonl -> {query_id: query_text}"""
    queries_file = os.path.join(dataset_path, "queries.jsonl")
    queries = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[str(q["_id"])] = q.get("text", "")
    return queries


def load_qrels(dataset_path: str, split: str = "test") -> Dict[str, Dict[str, int]]:
    """Loads qrels/{split}.tsv -> {query_id: {doc_id: relevance_score}}"""
    qrels_file = os.path.join(dataset_path, "qrels", f"{split}.tsv")
    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        # Check if first line is a header
        if header and header[0].lower() in ["query-id", "query_id"]:
            pass
        else:
            if header:
                qid, did, score = header[0], header[1], int(header[2])
                qrels.setdefault(qid, {})[did] = score

        for row in reader:
            if len(row) >= 3:
                qid, did, score = str(row[0]), str(row[1]), int(row[2])
                qrels.setdefault(qid, {})[did] = score
    return qrels
