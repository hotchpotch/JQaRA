import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base_reranker import BaseReranker


class SpladeReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_fp16=True,
        max_seq_length=512,
        batch_size=16,
    ):
        device = self._detect_device(device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if use_fp16 and "cuda" in device:
            self.model.half()

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _compute_vector(self, texts):
        device = self.model.device
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        logits, attention_mask = output.logits, tokens["attention_mask"]

        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vecs = max_val
        return vecs, tokens

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_emb, _ = self._compute_vector([query])
        query_emb = query_emb.squeeze(0)

        scores = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            document_embs, _ = self._compute_vector(batch)
            batch_scores = F.cosine_similarity(query_emb.unsqueeze(0), document_embs)
            scores.extend(batch_scores.tolist())

        return scores
