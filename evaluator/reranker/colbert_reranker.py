import torch
from transformers import AutoModel, AutoTokenizer

from .base_reranker import BaseReranker


class ColbertReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_fp16=True,
        max_length=512,
    ):
        device = self._detect_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        if use_fp16:
            self.model.half()
        self.model.eval()
        self.model.max_length = max_length
        self.max_length = max_length

    # base from: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-colbert-rerank/llama_index/postprocessor/colbert_rerank/base.py
    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_encoding = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.max_length,
            truncation="longest_first",
        )
        query_encoding = {
            key: value.to(self.device) for key, value in query_encoding.items()
        }
        with torch.no_grad():
            query_embedding = self.model(**query_encoding).last_hidden_state
        rerank_score_list = []

        for document_text in documents:
            document_encoding = self.tokenizer(
                document_text,
                return_tensors="pt",
                truncation="longest_first",
                max_length=self.max_length,
            )
            document_encoding = {
                key: value.to(self.device) for key, value in document_encoding.items()
            }
            with torch.no_grad():
                document_embedding = self.model(**document_encoding).last_hidden_state

            sim_matrix = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(2), document_embedding.unsqueeze(1), dim=-1
            )

            max_sim_scores, _ = torch.max(sim_matrix, dim=2)
            rerank_score_list.append(torch.mean(max_sim_scores, dim=1))
        sorted_scores = torch.stack(rerank_score_list).cpu().numpy()
        sorted_scores = sorted_scores.flatten().tolist()

        return sorted_scores
