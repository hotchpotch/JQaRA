import json
import os
import re
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk  # type: ignore
from ranx import Qrels, Run, compare, evaluate
from tqdm import tqdm

from reranker_factory import reranker_factory

from .logger import get_logger

DATASET_NAME = "hotchpotch/JQaRA"
DATASET_SPLIT = "test"

LOGGER = get_logger()


def _load_ds():
    return load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def _load_df():
    df = _load_ds.to_pandas()  # type: ignore
    df_q_id = df.groupby(["q_id", "question"]).agg(  # type: ignore
        {"passage_id": list, "label": list, "text": list, "title": list}
    )
    return df_q_id


def _pick_qrels(df: pd.DataFrame) -> Qrels:
    qrel_dict = {}
    for group_keys, (p_ids, labels, texts, titles) in tqdm(
        df.iterrows(), total=len(df), desc="pick qrels"
    ):
        LOGGER.debug("Pick qrels")
        q_id, question = group_keys  # type: ignore
        combine_passage_labels = dict(zip(p_ids, labels))
        # labels == 1 のものだけを取り出す
        combine_passage_labels = {
            k: v for k, v in combine_passage_labels.items() if v == 1
        }
        qrel_dict[q_id] = combine_passage_labels
    return Qrels(qrel_dict)


def _qrels(df: pd.DataFrame, cache_path: Path | None = None) -> Qrels:
    if cache_path is not None:
        qrels_file_name = "qrels"
        qrels_file_path = cache_path / f"qrels/{qrels_file_name}.lz4"
        if qrels_file_path.exists():
            LOGGER.debug(f"Load qrels from cache: {qrels_file_path}")
            return Qrels.from_file(str(qrels_file_path))
    qrel = _pick_qrels(df)
    if cache_path is not None:
        LOGGER.debug(f"Save qrels to cache: {qrels_file_path}")
        qrel.save(str(qrels_file_path))
    return qrel


def _run_rerank(
    reranker_name: str, run_name, df: pd.DataFrame, without_title: bool = False
) -> Run:
    LOGGER.debug(f"Run: {reranker_name}")
    reranker = reranker_factory(reranker_name)
    LOGGER.debug(f"- Reranker: {reranker}")

    run_dict = {}
    for group_keys, (passage_id, labels, texts, titles) in tqdm(
        df.iterrows(), total=len(df), desc=reranker_name
    ):
        q_id, question = group_keys  # type: ignore
        if not without_title:
            texts = [f"{title} {text}" for title, text in zip(titles, texts)]
        reranked_index, scores = reranker.rerank(question, texts)
        run_dict[q_id] = dict(zip(passage_id, scores))
    run = Run(run_dict, name=run_name)
    del reranker
    torch.cuda.empty_cache()
    return run


def _run(
    reranker_name: str,
    df: pd.DataFrame,
    without_title: bool = False,
    cache_path: Path | None = None,
) -> Run:
    if Path(reranker_name).exists():
        reranker_path = Path(reranker_name)
        reranker_name = str(reranker_path)

    run_name = re.sub(r"/+$", "", reranker_name).split("/")[-1]

    if cache_path is not None:
        runs_file_name = f"{run_name}"
        if without_title:
            runs_file_name = f"{runs_file_name}_without_title"
        runs_file_path = cache_path / f"runs/{runs_file_name}.lz4"
        if runs_file_path.exists():
            LOGGER.debug(f"Load run from cache: {runs_file_path}")
            return Run.from_file(str(runs_file_path), name=run_name)

    run = _run_rerank(
        reranker_name=reranker_name,
        df=df,
        run_name=run_name,
        without_title=without_title,
    )
    if cache_path is not None:
        LOGGER.debug(f"Save run to cache: {runs_file_path}")
        run.save(str(runs_file_path))
    return run


def runner(
    reranker_names: list[str],
    without_title: bool,
    cache_path: Path | None = None,
    n_samples: int | None = None,
):
    df = _load_df()
    if n_samples is not None:
        df = df.head(n_samples)
    qrel = _qrels(df, cache_path=cache_path)
    runs: list[Run] = []
    for reranker_name in reranker_names:
        run_result = _run(
            reranker_name=reranker_name,
            df=df,
            without_title=without_title,
            cache_path=cache_path,
        )
        runs.append(run_result)

    return qrel, runs
