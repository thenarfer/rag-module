# rag_module/core/retriever.py

from rag_module.core.store import search


def retrieve_top_k(query: str, top_k: int = 3):
    """
    Retrieve top_k chunks using current store.
    Returns list of dicts with text and score.
    """
    return search(query, top_k)
