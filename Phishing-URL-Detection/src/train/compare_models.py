"""Compare RF, Logistic Regression, and KNN on shared Validation only."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.evaluation import evaluate_classifier
from src.feature_extraction import FEATURE_COLUMNS
from src.train._common import (
    DEFAULT_PARTITIONS_DIR,
    PROJECT_ROOT,
    SharedPartitions,
    load_shared_partitions,
    write_comparison_report,
)


DEFAULT_RF_ARTIFACT = PROJECT_ROOT / "models" / "random_forest_candidate.joblib"
DEFAULT_LOGISTIC_ARTIFACT = PROJECT_ROOT / "models" / "logistic_regression.joblib"
DEFAULT_KNN_ARTIFACT = PROJECT_ROOT / "models" / "knn.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "model_comparison.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--partitions-dir",
        type=Path,
        default=DEFAULT_PARTITIONS_DIR,
        help="Shared Fit/Validation partition directory (default: %(default)s)",
    )
    parser.add_argument("--rf-artifact", default=str(DEFAULT_RF_ARTIFACT))
    parser.add_argument("--logistic-artifact", default=str(DEFAULT_LOGISTIC_ARTIFACT))
    parser.add_argument("--knn-artifact", default=str(DEFAULT_KNN_ARTIFACT))
    parser.add_argument("--output-report", default=str(DEFAULT_REPORT))
    return parser


def _artifact_error(path: str, field: str, expected, actual) -> ValueError:
    return ValueError(
        f"Incompatible artifact {path}: field {field!r} does not match shared partitions. "
        f"Expected: {expected!r}; actual: {actual!r}."
    )


def _load_and_validate_artifact(
    path: str,
    expected_name: str,
    partitions: SharedPartitions,
) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Unsupported artifact structure: {path}")
    if artifact.get("artifact_version") != 2:
        raise ValueError(f"Artifact {path} must use artifact_version 2.")
    if artifact.get("model_name") != expected_name:
        raise ValueError(
            f"Unexpected model in {path}: expected {expected_name!r}; "
            f"actual {artifact.get('model_name')!r}."
        )
    if "model" not in artifact:
        raise ValueError(f"Artifact {path} is missing its model.")

    expected_partition = {
        "fit_sha256": partitions.fit_sha256,
        "validation_sha256": partitions.validation_sha256,
        "partition_manifest_sha256": partitions.partition_manifest_sha256,
        "fit_rows": len(partitions.x_fit),
        "validation_rows": len(partitions.x_validation),
    }
    actual_partition = artifact.get("partition")
    if not isinstance(actual_partition, dict):
        raise _artifact_error(path, "partition", expected_partition, actual_partition)
    for field, expected in expected_partition.items():
        actual = actual_partition.get(field)
        if field.endswith("sha256"):
            matches = isinstance(actual, str) and actual.lower() == expected.lower()
        else:
            matches = actual == expected
        if not matches:
            raise _artifact_error(path, f"partition.{field}", expected, actual)

    actual_columns = artifact.get("feature_columns")
    if actual_columns != list(FEATURE_COLUMNS):
        raise _artifact_error(path, "feature_columns", list(FEATURE_COLUMNS), actual_columns)
    actual_feature_hash = artifact.get("feature_extraction_sha256")
    if (
        not isinstance(actual_feature_hash, str)
        or actual_feature_hash.lower() != partitions.feature_extraction_sha256.lower()
    ):
        raise _artifact_error(
            path,
            "feature_extraction_sha256",
            partitions.feature_extraction_sha256,
            actual_feature_hash,
        )
    return artifact


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    partitions = load_shared_partitions(args.partitions_dir)

    artifact_specs = (
        ("random_forest", args.rf_artifact),
        ("logistic_regression", args.logistic_artifact),
        ("knn", args.knn_artifact),
    )
    artifacts = {
        name: _load_and_validate_artifact(path, name, partitions)
        for name, path in artifact_specs
    }

    results = {}
    for name, _path in artifact_specs:
        artifact = artifacts[name]
        results[name] = evaluate_classifier(
            artifact["model"],
            partitions.x_validation.tolist(),
            partitions.y_validation,
            f"{name}_validation",
            fit_time_seconds=artifact.get("metrics", {}).get("training_time_seconds"),
            save_plot=False,
        )

    report = {
        "models": [name for name, _path in artifact_specs],
        "validation_rows": len(partitions.x_validation),
        "partition": {
            "fit_sha256": partitions.fit_sha256,
            "validation_sha256": partitions.validation_sha256,
            "partition_manifest_sha256": partitions.partition_manifest_sha256,
        },
        "feature_extraction_sha256": partitions.feature_extraction_sha256,
        "validation_metrics": results,
    }
    write_comparison_report(report, args.output_report)
    print(f"Saved comparison: {args.output_report}")
    return report


if __name__ == "__main__":
    main()
