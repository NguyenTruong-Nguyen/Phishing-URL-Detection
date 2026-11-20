"""Dataset loading, label normalization and deterministic splitting."""
from __future__ import annotations

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

POSITIVE_LABELS = {"1", "phishing", "malicious", "bad", "fraud", "unsafe"}
NEGATIVE_LABELS = {"0", "legitimate", "benign", "good", "safe", "normal"}


def normalize_label(value) -> int:
    """Map common label representations to 1=phishing, 0=legitimate.

    Inspect the source dataset documentation before relying on this mapping.
    """
    key = str(value).strip().lower()
    if key in POSITIVE_LABELS:
        return 1
    if key in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unknown label {value!r}. Update normalize_label after checking dataset documentation.")


def load_dataset(path: str, url_col: str = "URL", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (url_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s): {missing}. Available columns: {list(df.columns)}. "
            "Pass the correct --url-col and --label-col arguments."
        )
    result = df[[url_col, label_col]].copy()
    result.columns = ["url", "label_raw"]
    result["url"] = result["url"].astype(str).str.strip()
    result = result.dropna(subset=["label_raw"])
    result = result[result["url"].ne("")]
    result["label"] = result["label_raw"].map(normalize_label)
    result = result.drop_duplicates(subset=["url", "label"])
    if result["label"].nunique() != 2:
        raise ValueError("Dataset must contain both classes after label normalization.")
    return result[["url", "label"]].reset_index(drop=True)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        df["url"], df["label"], test_size=test_size,
        random_state=random_state, stratify=df["label"],
    )


def add_common_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data", default="data/raw/dataset.csv", help="Input CSV path")
    parser.add_argument("--url-col", default="URL", help="Column containing URL strings")
    parser.add_argument("--label-col", default="label", help="Column containing class labels")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser
