from abc import ABC, abstractmethod

import torch


def _auto_detect_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


class BaseReranker(ABC):
    def _detect_device(self, device: str) -> str:
        if device == "auto":
            device = _auto_detect_device()
        return device

    def rerank(self, query: str, documents: list[str]) -> tuple[list[int], list[float]]:
        scores = self._rerank(query, documents)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)
        return indices, scores

    @abstractmethod
    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        pass
