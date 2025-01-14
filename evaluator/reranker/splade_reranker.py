import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base_reranker import BaseReranker


def splade_max_pooling(logits, attention_mask):
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    return max_val


class SpladeReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_fp16=True,
        batch_size=16,
        query_max_length=512,
        document_max_length=512,
    ):
        device = self._detect_device(device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if use_fp16 and "cuda" in device:
            self.model.half()

        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.batch_size = batch_size

    def _compute_vector(self, texts, max_length: int):
        device = self.model.device
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        logits, attention_mask = output.logits, tokens["attention_mask"]

        return splade_max_pooling(logits, attention_mask)

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_emb = self._compute_vector([query], max_length=self.query_max_length)[0]
        doc_embs = []
        for i in range(0, len(documents), self.batch_size):
            doc_embs.append(
                self._compute_vector(
                    documents[i : i + self.batch_size],
                    max_length=self.document_max_length,
                )
            )
        doc_embs = torch.cat(doc_embs, dim=0)

        # scores = F.cosine_similarity(query_emb.unsqueeze(0), doc_embs)
        scores = torch.matmul(query_emb.unsqueeze(0), doc_embs.t()).squeeze(0)

        return scores.tolist()
