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


def report_to_table(report: Report, reranker_names: list[str]) -> str:
    return report.to_table()


def report_to_markdown(report: Report, reranker_names: list[str]) -> str:
    df = report_to_dataframe(report)
    return df.to_markdown(index=False)


def report_to_latex(report: Report, reranker_names: list[str]) -> str:
    return report.to_latex()


def report_to_csv(report: Report, reranker_names: list[str]) -> str:
    df = report_to_dataframe(report)
    return df.to_csv(index=False)


def report_to_markdown_with_links(report: Report, reranker_names: list[str]) -> str:
    df = report_to_dataframe(report)

    def reranker_name_to_link(name: str) -> str:
        if "text-embedding-" in name:
            return f"[{name}](https://platform.openai.com/docs/guides/embeddings)"
        elif "bm25" == name:
            return name
        else:
            hf_name = name.split("+")[0]  # BAAI/bge-m3+all -> BAAI/bge-m3
            name = name.split("/")[-1]  # BAAI/bge-m3+all -> bge-m3+all
            hf_url = f"https://huggingface.co/{hf_name}"
            return f"[{name}]({hf_url})"

    reranker_name_with_links = [reranker_name_to_link(name) for name in reranker_names]
    df["model_names"] = reranker_name_with_links
    return df.to_markdown(index=False)


reporters = {
    "table": report_to_table,
    "markdown": report_to_markdown,
    "latex": report_to_latex,
    "csv": report_to_csv,
    "markdown_with_links": report_to_markdown_with_links,
}
