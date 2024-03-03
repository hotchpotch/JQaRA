import io

import pandas as pd
from ranx import Qrels, Run, compare
from ranx.data_structures import Report


def compare_report(
    qrels: Qrels, runs: list[Run], metrics: list[str], max_p=0.01
) -> Report:
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=metrics,
        max_p=max_p,
    )
    return report


def report_to_dataframe(report: Report, round_digits: int | None = 4) -> pd.DataFrame:
    report_dict = report.to_dict()
    report_scores = {
        name: report_dict[name]["scores"] for name in report_dict["model_names"]
    }
    df = pd.DataFrame.from_dict(report_scores, orient="index")
    df = df.reset_index().rename(
        columns={"index": "model_names"}
    )  # index to model_names column
    if round_digits is not None:
        df = df.apply(
            lambda col: col.round(round_digits) if col.dtype == float else col
        )

    return df


def report_to_table(report: Report) -> str:
    return report.to_table()


def report_to_markdown(report: Report) -> str:
    df = report_to_dataframe(report)
    return df.to_markdown(index=False)


def report_to_latex(report: Report) -> str:
    return report.to_latex()


def report_to_csv(report: Report) -> str:
    df = report_to_dataframe(report)
    return df.to_csv(index=False)


reporters = {
    "table": report_to_table,
    "markdown": report_to_markdown,
    "latex": report_to_latex,
    "csv": report_to_csv,
    "dataframe": report_to_dataframe,
}
