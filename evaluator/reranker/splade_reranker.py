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
    ):
        device = self._detect_device(device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if use_fp16 and "cuda" in device:
            self.model.half()

        self.max_seq_length = max_seq_length

    def _compute_vector(self, text):
        device = self.model.device
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        logits, attention_mask = output.logits, tokens["attention_mask"]  # type: ignore

        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()
        return vec, tokens

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        # 毎度計算しているので遅い, batch で処理すべき
        query_emb, _ = self._compute_vector(query)
        document_embs = [self._compute_vector(doc)[0] for doc in documents]
        # to pt tensor
        document_embs = torch.stack(document_embs)
        scores = F.cosine_similarity(query_emb, document_embs).tolist()
        return scores
