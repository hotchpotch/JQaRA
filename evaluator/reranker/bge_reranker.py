from FlagEmbedding import FlagReranker

from .base_reranker import BaseReranker


class BgeReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_fp16=True,
        max_length=512,
    ):
        device = self._detect_device(device)
        print("Flag!")
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16,
        )
        if self.model.device != device:
            self.model.model.to(device)
        self.max_length = max_length

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        pairs: list[tuple[str, str]] = [(query, doc) for doc in documents]
        return self.model.compute_score(
            pairs, max_length=self.max_length, normalize=True
        )
