from FlagEmbedding import BGEM3FlagModel

from .base_reranker import BaseReranker


class BGEM3Reranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        targets=[("colbert", "0.4"), ("sparse", "0.2"), ("dense", "0.4")],
        use_fp16=True,
        batch_size=8,
        max_passage_length=512,
        max_query_length=128,
    ):
        device = self._detect_device(device)
        self.model = BGEM3FlagModel(
            model_name,
            device=device,
            use_fp16=use_fp16,
        )
        if len(targets) == 1:
            self.weights = [1.0, 1.0, 1.0]
        else:
            self.weights = [float(w) for _, w in targets]
        self.target_name: str = "+".join([name for name, _ in targets])
        self.batch_size = batch_size
        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        max_passage_length = self.max_passage_length
        max_query_length = self.max_query_length
        batch_size = self.batch_size

        sentence_pairs = [(query, doc) for doc in documents]
        scores = self.model.compute_score(
            sentence_pairs=sentence_pairs,
            weights_for_different_modes=self.weights,
            batch_size=batch_size,
            max_passage_length=max_passage_length,
            max_query_length=max_query_length,
        )[self.target_name]
        return scores
