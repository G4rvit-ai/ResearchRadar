"""Efficient sparse cosine similarity operations."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def ranked_search(index, query: str, top_k: int = 10) -> list[tuple[int, float]]:
    if not query or not query.strip():
        return []
    scores = cosine_similarity(index.transform_query(query), index.matrix).ravel()
    top_k = min(max(1, top_k), len(scores))
    candidates = np.argpartition(-scores, top_k - 1)[:top_k]
    return sorted(((int(i), float(scores[i])) for i in candidates), key=lambda item: -item[1])


def similar_documents(index, document_index: int, top_k: int = 10) -> list[tuple[int, float]]:
    if document_index < 0 or document_index >= index.matrix.shape[0]:
        raise IndexError("Document index is outside the TF-IDF matrix.")
    scores = cosine_similarity(index.matrix[document_index], index.matrix).ravel()
    scores[document_index] = -1.0
    count = min(max(1, top_k), max(1, len(scores) - 1))
    candidates = np.argpartition(-scores, count - 1)[:count]
    return sorted(((int(i), float(scores[i])) for i in candidates if i != document_index),
                  key=lambda item: -item[1])
