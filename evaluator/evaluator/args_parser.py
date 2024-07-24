from argparse import ArgumentParser
from typing import Any

DEFAULT_METRICS = "ndcg@10,mrr@10,ndcg@100,mrr@100"


def process_unknown_args(unknown_args: list[str]) -> dict[str, Any]:
    """
    未知の引数（kwargs）を処理する関数

    :param unknown_args: argparseで認識されなかった引数のリスト
    :return: 処理されたkwargsの辞書
    """
    kwargs: dict[str, Any] = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                key, value = key.split("=", 1)
            elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("-"):
                value = unknown_args[i + 1]
                i += 1
            else:
                value = True
            kwargs[key] = value
        i += 1
    return kwargs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-m",
        "--models",
        help="retriever / reranker model name",
        nargs="+",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--report_metrics",
        type=str,
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--output_format",
        help="output format, choose from table, markdown, csv, latex",
        type=str,
        choices=["table", "markdown", "csv", "latex", "markdown_with_links"],
        default="table",
    )
    parser.add_argument(
        "--max_p_value",
        type=float,
        default=0.01,
        help="Set the maximum p-value threshold for the statistical test, used only when the output format is 'table'.",
    )

    # 固定の引数をパースする
    known_args, unknown_args = parser.parse_known_args()

    # 未知の引数（kwargs）を処理する
    kwargs = process_unknown_args(unknown_args)

    # 固定の引数と動的な引数（kwargs）を結合
    all_args = vars(known_args)
    all_args["kwargs"] = kwargs

    return all_args
