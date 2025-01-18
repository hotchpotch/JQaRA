import torch
from sentence_transformers import SentenceTransformer

from .base_reranker import BaseReranker


class SentenceTransformerReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        query_prefix: str = "",
        document_prefix: str = "",
        use_fp16=True,
        normalize_embeddings=True,
        max_seq_length=512,
    ):
        device = self._detect_device(device)
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )
        if use_fp16 and "cuda" in device:
            self.model.half()

        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize_embeddings = normalize_embeddings
        try:
            self.model.max_seq_length = max_seq_length
        except AttributeError as e:
            print("max_seq_length is not supported for this model")
            pass

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_prefix = self.query_prefix
        document_prefix = self.document_prefix
        query_text = query_prefix + query
        documents = [document_prefix + doc for doc in documents]
        embs = self.model.encode(
            [query_text] + documents, normalize_embeddings=True, convert_to_tensor=True
        )
        query_emb = embs[0]
        document_embs = embs[1:]
        scores = torch.nn.functional.cosine_similarity(
            query_emb,
            document_embs,  # type: ignore
        ).tolist()
        return scores
