import numpy as np
from langchain_openai import OpenAIEmbeddings

from .base_reranker import BaseReranker


class OpenAIEmbeddingsReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        diminsions: int | None = None,
    ):
        self.model = OpenAIEmbeddings(
            model=model_name, tiktoken_model_name="cl100k_base", dimensions=diminsions
        )

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_docs = [query] + documents
        embs = self.model.embed_documents(query_docs)
        embs = np.array(embs)
        query_emb = embs[0]
        document_embs = embs[1:]
        scores = query_emb.dot(document_embs.T)
        return scores.tolist()
