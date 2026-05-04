import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi


def load_index(
    collection_name: str = "rag_docs",
    persist_dir: str = "./chroma_db",
    bm25_path: str = "./bm25_index.pkl",
) -> Tuple[chromadb.Collection, BM25Okapi, List[str], List[Dict]]:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_collection(collection_name)
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return collection, data["bm25"], data["chunks"], data["metadatas"]


def _embed_query(query: str, client: OpenAI, model: str = "text-embedding-3-small") -> List[float]:
    response = client.embeddings.create(input=[query], model=model)
    return response.data[0].embedding


def dense_retrieve(
    query: str,
    collection: chromadb.Collection,
    client: OpenAI,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    query_embedding = _embed_query(query, client)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        output.append(
            {"text": doc, "metadata": meta, "score": float(1 - dist), "retrieval_type": "dense"}
        )
    return output


def sparse_retrieve(
    query: str,
    bm25: BM25Okapi,
    chunks: List[str],
    metadatas: List[Dict],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i], "metadata": metadatas[i], "score": float(scores[i]), "retrieval_type": "sparse"}
        for i in top_indices
        if scores[i] > 0
    ]


def hybrid_retrieve(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    chunks: List[str],
    metadatas: List[Dict],
    client: OpenAI,
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of dense + sparse results."""
    dense = dense_retrieve(query, collection, client, top_k=top_k * 2)
    sparse = sparse_retrieve(query, bm25, chunks, metadatas, top_k=top_k * 2)

    rrf_scores: Dict[str, float] = {}
    text_to_result: Dict[str, Dict] = {}

    for rank, result in enumerate(dense):
        key = result["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        text_to_result[key] = result

    for rank, result in enumerate(sparse):
        key = result["text"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if key not in text_to_result:
            text_to_result[key] = result

    sorted_texts = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)[:top_k]
    output = []
    for text in sorted_texts:
        result = text_to_result[text].copy()
        result["score"] = rrf_scores[text]
        result["retrieval_type"] = "hybrid"
        output.append(result)
    return output
