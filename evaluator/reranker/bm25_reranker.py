from rank_bm25 import BM25Okapi

from .base_reranker import BaseReranker


def create_text_splitter_fn():
    from fugashi import Tagger  # type: ignore

    tagger = Tagger("-Owakati")

    def tokenize_split(text: str) -> list[str]:
        return tagger.parse(text).split(" ")

    return tokenize_split


class BM25Reranker(BaseReranker):
    def __init__(
        self,
        text_splitter_fn=None,
    ):
        if text_splitter_fn is None:
            text_splitter_fn = create_text_splitter_fn()
        self.text_splitter_fn = text_splitter_fn

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        tokenized_docs = list(map(self.text_splitter_fn, documents))
        tokenized_query = self.text_splitter_fn(query)
        bm25 = BM25Okapi(tokenized_docs)
        doc_scores = bm25.get_scores(tokenized_query).tolist()
        return doc_scores
