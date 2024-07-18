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
        return vecs

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        all_texts = [query] + documents
        all_embeddings = []

        for i in range(0, len(all_texts), self.batch_size):
            batch = all_texts[i : i + self.batch_size]
            batch_embeddings = self._compute_vector(batch)
            all_embeddings.append(batch_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        query_emb = all_embeddings[0]
        doc_embs = all_embeddings[1:]

        scores = F.cosine_similarity(query_emb.unsqueeze(0), doc_embs)
        return scores.tolist()
