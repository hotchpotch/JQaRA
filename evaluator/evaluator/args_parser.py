from argparse import ArgumentParser

DEFAULT_METRICS = "ndcg@10,mrr@10,ndcg@100,mrr@100"


def parse_args():
    args = ArgumentParser()
    args.add_argument("-d", "--debug", action="store_true")
    args.add_argument("-v", "--verbose", action="store_true")
    args.add_argument(
        "-m",
        "--models",
        help="retriever / reranker model name",
        nargs="+",
        type=list[str],
        default=None,
        required=True,
    )
    args.add_argument(
        "-r",
        "--report_metrics",
        type=str,
        default=DEFAULT_METRICS,
    )
    args.add_argument(
        "--no_cache",
        action="store_true",
    )
    args.add_argument(
        "-o",
        "--output_csv",
        help="output csv file path, example: ./report.csv",
        type=str,
        default=None,
    )
    args.add_argument(
        "--max_p_value",
        type=float,
        default=0.01,
        help="max p value for fisher test",
    )
    args.add_argument("--without_title", action="store_true")
    parsed_args = args.parse_args()
    return parsed_args
