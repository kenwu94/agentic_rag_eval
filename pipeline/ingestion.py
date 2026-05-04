import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: List[str], client: OpenAI, model: str = "text-embedding-3-small") -> List[List[float]]:
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def ingest_documents(
    docs_dir: str,
    collection_name: str = "rag_docs",
    chunk_size: int = 400,
    overlap: int = 50,
    persist_dir: str = "./chroma_db",
    bm25_path: str = "./bm25_index.pkl",
) -> Tuple[chromadb.Collection, BM25Okapi, List[str], List[Dict]]:
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    all_chunks: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []

    for filepath in sorted(Path(docs_dir).glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append(
                {"source": filepath.name, "chunk_index": i, "doc_id": filepath.stem}
            )
            all_ids.append(f"{filepath.stem}_{i}")
        print(f"  Loaded {filepath.name}: {len(chunks)} chunks")

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks, client)

    collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas)

    bm25 = BM25Okapi([chunk.lower().split() for chunk in all_chunks])
    with open(bm25_path, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "chunks": all_chunks, "metadatas": all_metadatas, "ids": all_ids}, f
        )

    print(f"Ingestion complete: {len(all_chunks)} chunks indexed.")
    return collection, bm25, all_chunks, all_metadatas
