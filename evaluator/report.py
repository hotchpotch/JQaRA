import pandas as pd
from ranx import Qrels, Run, compare


def compare_report(qrels: Qrels, runs: list[Run], metrics: list[str], max_p=0.05):
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=metrics,
        max_p=max_p,
    )
    return report


def report_to_df(report):
    report_dict = report.to_dict()
    report_scores = {
        name: report_dict[name]["scores"] for name in report_dict["model_names"]
    }
    report_df = pd.DataFrame.from_dict(report_scores, orient="index")
    # index to name column
    report_df = report_df.rename_axis("name")
    return report_df
