import re
from pathlib import Path

import pandas as pd
import torch
from ranx import Qrels, Run
from tqdm import tqdm

from .logger import get_logger
from .reranker_factory import reranker_factory

LOGGER = get_logger()


def _pick_qrels(df: pd.DataFrame) -> Qrels:
    LOGGER.debug("Pick qrels")
    qrel_dict = {}
    for group_keys, (p_ids, labels, texts, titles) in tqdm(
        df.iterrows(), total=len(df), desc="pick qrels"
    ):
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
        qrels_file_path = cache_path / f"qrels/{qrels_file_name}.trec"
        if qrels_file_path.exists():
            LOGGER.debug(f"Load qrels from cache: {qrels_file_path}")
            return Qrels.from_file(str(qrels_file_path))
    qrel = _pick_qrels(df)
    if cache_path is not None:
        LOGGER.debug(f"Save qrels to cache: {qrels_file_path}")
        qrels_file_path.parent.mkdir(parents=True, exist_ok=True)
        qrel.save(str(qrels_file_path))
    return qrel


def _run_rerank(
    reranker_name: str,
    run_name,
    df: pd.DataFrame,
    without_title: bool = False,
    kwargs: dict = {},
) -> Run:
    LOGGER.debug(f"Run: {reranker_name}")
    reranker = reranker_factory(reranker_name, kwargs=kwargs)
    LOGGER.debug(f"- Reranker: {reranker.__class__}")

    run_dict = {}
    for group_keys, (p_id, labels, texts, titles) in tqdm(
        df.iterrows(), total=len(df), desc=reranker_name
    ):
        q_id, question = group_keys  # type: ignore
        if not without_title:
            texts = [f"{title} {text}" for title, text in zip(titles, texts)]
        reranked_index, scores = reranker.rerank(question, texts)
        run_dict[q_id] = dict(zip(p_id, scores))
    run = Run(run_dict, name=run_name)
    del reranker
    torch.cuda.empty_cache()
    return run


def _run(
    reranker_name: str,
    df: pd.DataFrame,
    without_title: bool = False,
    cache_path: Path | None = None,
    kwargs: dict = {},
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
        kwargs=kwargs,
    )
    if cache_path is not None:
        LOGGER.debug(f"Save run to cache: {runs_file_path}")
        runs_file_path.parent.mkdir(parents=True, exist_ok=True)
        run.save(str(runs_file_path))
    return run


def runner(
    reranker_names: list[str],
    df: pd.DataFrame,
    without_title: bool = False,
    cache_path: Path | None = None,
    kwargs: dict = {},
):
    qrel = _qrels(df, cache_path=cache_path)
    runs: list[Run] = []
    for reranker_name in reranker_names:
        run_result = _run(
            reranker_name=reranker_name,
            df=df,
            without_title=without_title,
            cache_path=cache_path,
            kwargs=kwargs,
        )
        runs.append(run_result)

    return qrel, runs
