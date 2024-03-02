import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from evaluator.args_parser import parse_args
from evaluator.load_data import load_df
from evaluator.logger import get_logger
from evaluator.reporter import compare_report, report_to_df
from evaluator.runner import runner


def main():
    parsed_args = parse_args()
    logger = get_logger()
    if parsed_args.verbose:
        logger.setLevel("DEBUG")
    n_samples = None
    if parsed_args.debug:
        n_samples = 100

    if parsed_args.no_cache:
        cache_path = None
    else:
        cache_path = Path("evaluator_results")
        if parsed_args.debug:
            cache_path = cache_path / "debug"

    reranker_names = []
    for reranker_name in parsed_args.models:
        if isinstance(reranker_name, list):
            reranker_name = "".join(reranker_name)
        reranker_names.append(reranker_name)

    df = load_df()
    logger.info(f"Load data: {len(df)}")
    if n_samples is not None:
        df = df.sample(n_samples)
        logger.info(f"Use {n_samples} samples for debug mode")

    qrel, runs = runner(
        reranker_names=reranker_names,
        df=df,
        without_title=parsed_args.without_title,
        cache_path=cache_path,
    )

    report_metrics = parsed_args.report_metrics.split(",")
    max_p = parsed_args.max_p_value
    report = compare_report(
        qrels=qrel,
        runs=runs,
        metrics=report_metrics,
        max_p=max_p,
    )
    print(report.to_table())

    csv_path = parsed_args.output_csv
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        report_df = report_to_df(report)
        report_df.to_csv(csv_path)
        logger.info(f"Save report to {csv_path}")


if __name__ == "__main__":
    main()
