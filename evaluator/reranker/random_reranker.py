import random

from .base_reranker import BaseReranker


class RandomReranker(BaseReranker):
    def __init__(
        self,
        seed=42,
    ):
        self.seed = seed

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        random.seed(self.seed)
        doc_scores = [random.random() for _ in documents]
        return doc_scores
