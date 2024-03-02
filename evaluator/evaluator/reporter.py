import pandas as pd
from ranx import Qrels, Run, compare


def compare_report(qrels: Qrels, runs: list[Run], metrics: list[str], max_p=0.01):
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=metrics,
        max_p=max_p,
    )
    return report


def report_to_df(report, round_digits: int | None = 4):
    report_dict = report.to_dict()
    report_scores = {
        name: report_dict[name]["scores"] for name in report_dict["model_names"]
    }
    report_df = pd.DataFrame.from_dict(report_scores, orient="index")
    # index to name column
    report_df = report_df.rename_axis("name")
    if round_digits is not None:
        report_df = report_df.apply(
            lambda col: col.round(round_digits) if col.dtype == float else col
        )
    return report_df
