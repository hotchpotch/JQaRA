from sentence_transformers import CrossEncoder

from .base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        default_activation_function=None,
        use_fp16=True,
        max_length=512,
    ):
        device = self._detect_device(device)
        self.model = CrossEncoder(
            model_name,
            device=device,
            default_activation_function=default_activation_function,
            trust_remote_code=True,
        )
        if use_fp16 and "cuda" in device:
            self.model.model.half()
        self.model.max_length = max_length

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        pairs: list[list[str]] = [[query, doc] for doc in documents]
        scores = list(map(float, self.model.predict(pairs)))
        return scores
