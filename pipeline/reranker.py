from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Score query-document pairs with a cross-encoder and return top_k by score."""
    if not results:
        return []
    model = _get_model()
    scores = model.predict([(query, r["text"]) for r in results])
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:top_k]
