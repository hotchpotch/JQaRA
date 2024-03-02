from datasets import load_dataset

DATASET_NAME = "hotchpotch/JQaRA"
DATASET_SPLIT = "test"


def load_ds():
    return load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def load_df():
    df = load_ds().to_pandas()  # type: ignore
    df_q_id = df.groupby(["q_id", "question"]).agg(  # type: ignore
        {"passage_row_id": list, "label": list, "text": list, "title": list}
    )
    return df_q_id
