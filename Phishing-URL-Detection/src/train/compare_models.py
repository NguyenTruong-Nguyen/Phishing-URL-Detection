"""Compare the three supported algorithms using registered-domain Validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.train._common import domain_split, write_comparison_report
from src.feature_extraction import FEATURE_COLUMNS
from src.preprocessing import add_common_cli_args, load_dataset


def main():
    parser = add_common_cli_args(argparse.ArgumentParser())
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--baseline-report", default="reports/baseline_metrics.json")
    parser.add_argument("--logistic-artifact", default="models/logistic_regression.joblib")
    parser.add_argument("--knn-artifact", default="models/knn.joblib")
    parser.add_argument("--output-report", default="reports/model_comparison.json")
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    data = load_dataset(args.data, args.url_col, args.label_col)
    _, x_validation, _, y_validation, split = domain_split(
        data, args.test_size, args.random_state, args.validation_size
    )
    for key in ("fit_rows", "validation_rows", "test_rows", "random_state"):
        if split[key] != baseline["split"].get(key):
            raise RuntimeError(f"Recreated split differs from baseline report: {key}.")

    rf_artifact = joblib.load("models/model_v0.joblib")
    if rf_artifact.get("feature_columns") != list(FEATURE_COLUMNS):
        raise RuntimeError("Random Forest artifact does not match the 24-feature schema.")
    results = {
        "random_forest": baseline["candidate_metrics"]["random_forest"]["validation"],
    }
    for name, artifact_path in (
        ("logistic_regression", args.logistic_artifact),
        ("knn", args.knn_artifact),
    ):
        artifact = joblib.load(artifact_path)
        if artifact["model_name"] != name:
            raise ValueError(f"Unexpected model in {artifact_path}: {artifact['model_name']}")
        if artifact.get("feature_columns") != list(FEATURE_COLUMNS):
            raise RuntimeError(f"{name} artifact does not match the shared 24-feature schema.")
        for key in (
            "fit_rows",
            "validation_rows",
            "test_rows",
            "random_state",
            "validation_random_state",
        ):
            if artifact["split"].get(key) != split[key]:
                raise RuntimeError(f"{name} was trained on a different split: {key}.")
        # Recompute on the shared Validation partition; neither artifact has
        # been fitted on it. This is selection-time validation, never Test.
        from src.evaluation import evaluate_classifier

        results[name] = evaluate_classifier(
            artifact["model"],
            x_validation.tolist(),
            y_validation,
            f"{name}_validation",
            fit_time_seconds=artifact.get("metrics", {}).get("training_time_seconds"),
            save_plot=False,
        )

    report = {
        "models": ["random_forest", "logistic_regression", "knn"],
        "validation_rows": len(x_validation),
        "split": split,
        "test_used_for_selection": False,
        "random_forest_metrics_source": "existing baseline_metrics.json fit-only validation result",
        "validation_metrics": results,
    }
    write_comparison_report(report, args.output_report)
    print(f"Saved comparison: {args.output_report}")
if __name__ == "__main__":
    main()
