# rag_module/core/store.py
from typing import List, Dict
from uuid import uuid4
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Persistent DB on disk (survives restarts)
_client = chromadb.PersistentClient(path="./chroma")
_embed = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
_col = _client.get_or_create_collection(name="rag_docs", embedding_function=_embed)

def add_document(text_chunks: List[str], metadata: Dict) -> str:
    doc_id = f"doc_{uuid4().hex[:8]}"
    ids = [f"{doc_id}_{i}" for i in range(len(text_chunks))]
    metas = [{**(metadata or {}), "doc_id": doc_id} for _ in text_chunks]
    _col.add(ids=ids, documents=text_chunks, metadatas=metas)
    return doc_id

def search(query: str, top_k: int = 3) -> List[Dict]:
    res = _col.query(query_texts=[query], n_results=top_k,
                     include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # smaller is better (cosine)
    out = []
    for text, meta, dist in zip(docs, metas, dists):
        score = float(1.0 - dist) if dist is not None else 0.0  # higher = more similar
        out.append({"doc_id": meta.get("doc_id", ""), "text": text, "metadata": meta, "score": score})
    # Already sorted by Chroma, but keep this to be safe
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]
