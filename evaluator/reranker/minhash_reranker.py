from datasketch import MinHash

from .base_reranker import BaseReranker


def create_text_splitter_fn():
    from fugashi import Tagger  # type: ignore

    tagger = Tagger("-Owakati")

    def tokenize_split(text: str) -> list[str]:
        return tagger.parse(text).split(" ")

    return tokenize_split


class MinHashReranker(BaseReranker):
    def __init__(
        self,
        text_splitter_fn=None,
    ):
        if text_splitter_fn is None:
            text_splitter_fn = create_text_splitter_fn()
        self.text_splitter_fn = text_splitter_fn

    def create_minhash(self, texts: list[str]) -> MinHash:
        minhash = MinHash()
        for text in texts:
            minhash.update(text.encode("utf-8"))
        return minhash

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        tokenized_docs = list(map(self.text_splitter_fn, documents))
        mimhash_docs = list(map(self.create_minhash, tokenized_docs))
        tokenized_query = self.text_splitter_fn(query)
        minhash_query = self.create_minhash(tokenized_query)
        doc_scores = [
            minhash_query.jaccard(mimhash_doc) for mimhash_doc in mimhash_docs
        ]
        return doc_scores
