"""Shared, registered-domain-isolated data and artifact helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.evaluation import evaluate_classifier
from src.feature_extraction import FEATURE_COLUMNS
from src.preprocessing import (
    LABEL_NAMES,
    PHIUSIIL_LABEL_MAPPING,
    add_common_cli_args,
    load_dataset,
    registered_domain,
    split_dataset,
)


def add_training_args(parser: argparse.ArgumentParser, default_output: str):
    add_common_cli_args(parser)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--output", default=default_output)
    return parser


def domain_split(data: pd.DataFrame, test_size: float, random_state: int, validation_size: float):
    """Reproduce the baseline outer train/test and inner fit/validation splits."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    x_train, x_test, y_train, y_test = split_dataset(
        data, test_size, random_state, strategy="domain"
    )
    train_frame = pd.DataFrame(
        {"url": x_train.to_numpy(), "label": y_train.to_numpy()}
    )
    x_fit, x_validation, y_fit, y_validation = split_dataset(
        train_frame,
        test_size=validation_size,
        random_state=random_state + 1,
        strategy="domain",
    )
    groups = {
        "fit": {registered_domain(str(url)) for url in x_fit},
        "validation": {registered_domain(str(url)) for url in x_validation},
        "test": {registered_domain(str(url)) for url in x_test},
    }
    overlap = {
        "fit_validation": len(groups["fit"] & groups["validation"]),
        "fit_test": len(groups["fit"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Registered-domain leakage detected: {overlap}")
    summary = {
        "outer_strategy": "registered_domain",
        "random_state": random_state,
        "validation_random_state": random_state + 1,
        "fit_rows": int(len(x_fit)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "fit_domains": len(groups["fit"]),
        "validation_domains": len(groups["validation"]),
        "test_domains": len(groups["test"]),
        "domain_overlap": overlap,
    }
    # x_test/y_test are intentionally not returned to training/evaluation code.
    return x_fit.reset_index(drop=True), x_validation.reset_index(drop=True), y_fit.reset_index(drop=True), y_validation.reset_index(drop=True), summary


def train_and_save(model, model_name: str, args) -> dict:
    """Fit on Fit only, score Validation, and create a non-overwriting artifact."""
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model artifact: {output}")
    data = load_dataset(args.data, args.url_col, args.label_col)
    x_fit, x_validation, y_fit, y_validation, split_info = domain_split(
        data, args.test_size, args.random_state, args.validation_size
    )
    start = time.perf_counter()
    model.fit(x_fit.tolist(), y_fit)
    fit_seconds = time.perf_counter() - start
    metrics = evaluate_classifier(
        model,
        x_validation.tolist(),
        y_validation,
        f"{model_name}_validation",
        fit_time_seconds=fit_seconds,
        save_plot=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 1,
        "model": model,
        "model_type": "raw_url_features",
        "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_mapping": {
            "source_to_internal": dict(PHIUSIIL_LABEL_MAPPING),
            "internal": {str(key): value for key, value in LABEL_NAMES.items()},
        },
        "training_config": {"fit_partition": "registered-domain Fit only"},
        "split": split_info,
        "metrics": metrics,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(artifact, output)
    loaded = joblib.load(output)
    probe = loaded["model"].predict(x_validation.iloc[:1].tolist())
    if len(probe) != 1:
        output.unlink(missing_ok=True)
        raise RuntimeError("Reloaded artifact failed its prediction check.")
    return {
        "model": model_name,
        "training_time_seconds": fit_seconds,
        "validation_metrics": metrics,
        "split": split_info,
        "artifact": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "artifact_bytes": output.stat().st_size,
    }


def write_comparison_report(report: dict, path: str = "reports/model_comparison.json"):
    """Write a new comparison report without overwriting existing results."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
