import torch

from reranker.bge_m3_reranker import BGEM3Reranker
from reranker.bge_reranker import BgeReranker
from reranker.bm25_reranker import BM25Reranker
from reranker.colbert_reranker import ColbertReranker
from reranker.cross_encoder_reranker import CrossEncoderReranker
from reranker.minhash_reranker import MinHashReranker
from reranker.openai_embeddings_reranker import OpenAIEmbeddingsReranker
from reranker.random_reranker import RandomReranker
from reranker.sentence_transformer_reranker import SentenceTransformerReranker
from reranker.splade_reranker import SpladeReranker


def reranker_factory(model_name: str, device: str = "auto", use_fp16=True, kwargs={}):
    if "bm25" == model_name:
        return BM25Reranker()
    elif "minhash" in model_name:
        return MinHashReranker()
    elif "random" in model_name:
        return RandomReranker()
    elif "text-embedding-" in model_name:
        return OpenAIEmbeddingsReranker(model_name)
    elif "-e5-" in model_name:
        if "+query" in model_name:
            model_name = model_name.replace("+query", "")
            query_prefix = "query: "
            document_prefix = "query: "
        else:
            query_prefix = "query: "
            document_prefix = "passage: "
        return SentenceTransformerReranker(
            model_name,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            use_fp16=use_fp16,
            device=device,
        )
    elif "fio-" in model_name:
        return SentenceTransformerReranker(
            model_name,
            query_prefix="関連記事を取得するために使用できるこの文の表現を生成します: ",
            use_fp16=use_fp16,
            device=device,
        )
    elif "bge-m3" in model_name:
        target_model_name = model_name.split("+")[0]
        arg = model_name.split("+")[1]
        if arg == "all":
            return BGEM3Reranker(
                target_model_name,
                device=device,
                use_fp16=use_fp16,
                targets=[("colbert", "0.4"), ("sparse", "0.2"), ("dense", "0.4")],
            )
        elif arg == "colbert":
            return BGEM3Reranker(
                target_model_name,
                device=device,
                use_fp16=use_fp16,
                targets=[("colbert", "1.0")],
            )
        elif arg == "sparse":
            return BGEM3Reranker(
                target_model_name,
                device=device,
                use_fp16=use_fp16,
                targets=[("sparse", "1.0")],
            )
        elif arg == "dense":
            return BGEM3Reranker(
                target_model_name,
                device=device,
                use_fp16=use_fp16,
                targets=[("dense", "1.0")],
            )
        else:
            # same dense
            return BGEM3Reranker(
                target_model_name,
                device=device,
                use_fp16=use_fp16,
                targets=[("dense", "1.0")],
            )
    elif "colbert" in model_name or "ColBERT" in model_name:
        return ColbertReranker(
            model_name,
            device=device,
            use_fp16=use_fp16,
        )
    elif "bge-reranker" in model_name:
        return BgeReranker(
            model_name,
            device=device,
            use_fp16=use_fp16,
        )
    elif "cl-nagoya/shioriha-large-reranker" in model_name:
        return CrossEncoderReranker(
            model_name,
            device=device,
            default_activation_function=torch.nn.Identity(),
            use_fp16=use_fp16,
        )
    elif "splade" in model_name:
        query_max_length = int(kwargs.get("query_max_length", 512))
        document_max_length = int(kwargs.get("document_max_length", 512))
        return SpladeReranker(
            model_name,
            device=device,
            use_fp16=use_fp16,
            query_max_length=query_max_length,
            document_max_length=document_max_length,
        )
    elif (
        "ce-" in model_name
        or "cross-encoder" in model_name
        or "exp" in model_name
        or "reranker" in model_name
    ):
        return CrossEncoderReranker(
            model_name,
            device=device,
            use_fp16=use_fp16,
        )
    else:
        return SentenceTransformerReranker(model_name, device=device, use_fp16=use_fp16)
