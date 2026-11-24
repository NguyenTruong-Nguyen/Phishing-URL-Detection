"""Shared, registered-domain-isolated data and artifact helpers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.evaluation import evaluate_classifier
from src.feature_extraction import FEATURE_COLUMNS
from src.preprocessing import (
    LABEL_NAMES,
    PHIUSIIL_LABEL_MAPPING,
    registered_domain,
    split_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTITIONS_DIR = PROJECT_ROOT / "data" / "partitions" / "shared"
_PARTITION_FILES = {"fit": "fit.csv", "validation": "validation.csv"}
_WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}"
    for prefix in ("com", "lpt")
    for number in range(1, 10)
}


class SharedPartitionError(ValueError):
    """Raised when shared Fit/Validation data fail integrity checks."""


@dataclass(frozen=True)
class SharedPartitions:
    x_fit: pd.Series
    y_fit: pd.Series
    x_validation: pd.Series
    y_validation: pd.Series
    fit_sha256: str
    validation_sha256: str
    partition_manifest_sha256: str
    feature_extraction_sha256: str
    manifest: dict


def _safe_partition_filename(value: object, expected: str) -> str:
    if not isinstance(value, str) or not value:
        raise SharedPartitionError("Manifest partition path must be a non-empty filename.")
    windows_path = PureWindowsPath(value)
    stem = windows_path.stem.casefold()
    if (
        windows_path.drive
        or windows_path.root
        or windows_path.name != value
        or "/" in value
        or "\\" in value
        or ".." in windows_path.parts
        or stem in _WINDOWS_DEVICE_NAMES
    ):
        raise SharedPartitionError(f"Unsafe manifest partition filename: {value!r}.")
    if value != expected:
        raise SharedPartitionError(
            f"Manifest must reference exactly {expected!r}; received {value!r}."
        )
    return value


def _canonical_manifest_sha256(manifest: dict) -> str:
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalized_feature_extraction_sha256() -> str:
    source = PROJECT_ROOT / "src" / "feature_extraction.py"
    content = source.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _read_partition_csv(content: bytes, expected_rows: object, partition_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            dtype={"url": "string"},
            keep_default_na=False,
            encoding="utf-8",
        )
    except Exception as exc:
        raise SharedPartitionError(f"Could not parse {partition_name} partition CSV.") from exc

    if list(frame.columns) != ["url", "label"]:
        raise SharedPartitionError(
            f"{partition_name} partition must contain exactly the columns url,label in that order."
        )
    try:
        csv_rows = list(csv.reader(io.StringIO(content.decode("utf-8"), newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise SharedPartitionError(f"{partition_name} partition is not valid UTF-8 CSV.") from exc
    if (
        not csv_rows
        or csv_rows[0] != ["url", "label"]
        or len(csv_rows) - 1 != len(frame)
        or any(len(row) != 2 or row[1] not in {"0", "1"} for row in csv_rows[1:])
    ):
        raise SharedPartitionError(
            f"{partition_name} labels must be CSV values exactly equal to 0 or 1."
        )
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
        raise SharedPartitionError(f"Manifest {partition_name} row count must be a positive integer.")
    if len(frame) != expected_rows:
        raise SharedPartitionError(
            f"{partition_name} partition row count mismatch. Expected: {expected_rows}; actual: {len(frame)}."
        )
    if frame["url"].isna().any() or frame["url"].str.strip().eq("").any():
        raise SharedPartitionError(f"{partition_name} partition contains an empty URL.")
    if frame["url"].duplicated().any():
        raise SharedPartitionError(f"{partition_name} partition contains duplicate URLs.")

    valid_labels = frame["label"].map(
        lambda value: isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) in (0, 1)
    )
    if not valid_labels.all():
        raise SharedPartitionError(
            f"{partition_name} labels must be integer values exactly equal to 0 or 1."
        )
    frame["label"] = frame["label"].astype("int64")
    if set(frame["label"].unique()) != {0, 1}:
        raise SharedPartitionError(f"{partition_name} partition must contain both labels 0 and 1.")
    return frame


def load_shared_partitions(partitions_dir: str | Path = DEFAULT_PARTITIONS_DIR) -> SharedPartitions:
    """Load and validate only the shared Fit and Validation files."""
    directory = Path(partitions_dir).expanduser().resolve()
    manifest_path = directory / "split_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SharedPartitionError(f"Shared partition manifest does not exist: {manifest_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SharedPartitionError(f"Could not read a valid shared partition manifest: {manifest_path}") from exc

    if (
        not isinstance(manifest, dict)
        or isinstance(manifest.get("manifest_version"), bool)
        or not isinstance(manifest.get("manifest_version"), int)
        or manifest.get("manifest_version") != 1
    ):
        raise SharedPartitionError("Shared partition manifest has an unsupported schema/version.")
    partition_section = manifest.get("partitions")
    if not isinstance(partition_section, dict):
        raise SharedPartitionError("Shared partition manifest is missing the partitions object.")

    metadata_by_name = {}
    paths = {}
    raw_hashes = {}
    raw_contents = {}
    for name, expected_filename in _PARTITION_FILES.items():
        item = partition_section.get(name)
        if not isinstance(item, dict):
            raise SharedPartitionError(f"Shared partition manifest is missing {name} metadata.")
        filename = _safe_partition_filename(item.get("path"), expected_filename)
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise SharedPartitionError(f"Manifest {name} SHA-256 must be a 64-character hexadecimal string.")
        path = directory / filename
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SharedPartitionError(f"Could not read {name} partition CSV: {path}") from exc
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash.lower() != expected_hash.lower():
            raise SharedPartitionError(
                f"{name.title()} partition SHA-256 mismatch.\n"
                f"Expected: {expected_hash}\nActual: {actual_hash}\nTraining has been stopped."
            )
        metadata_by_name[name] = item
        paths[name] = path
        raw_hashes[name] = actual_hash
        raw_contents[name] = raw

    manifest_features = manifest.get("features")
    if (
        not isinstance(manifest_features, dict)
        or manifest_features.get("count") != len(FEATURE_COLUMNS)
        or manifest_features.get("names") != list(FEATURE_COLUMNS)
    ):
        raise SharedPartitionError("Manifest feature names/count do not match the current feature extraction.")
    expected_mapping = {
        "source_to_internal": dict(PHIUSIIL_LABEL_MAPPING),
        "internal": {str(label): name for label, name in LABEL_NAMES.items()},
    }
    if manifest.get("label_mapping") != expected_mapping:
        raise SharedPartitionError("Manifest label mapping does not match the project mapping.")
    manifest_split = manifest.get("split")
    if (
        not isinstance(manifest_split, dict)
        or not isinstance(manifest_split.get("strategy"), str)
        or isinstance(manifest_split.get("inner_random_state"), bool)
        or not isinstance(manifest_split.get("inner_random_state"), int)
    ):
        raise SharedPartitionError("Manifest split metadata is missing the shared strategy or inner seed.")

    frames = {
        name: _read_partition_csv(
            raw_contents[name], metadata_by_name[name].get("rows"), name
        )
        for name in _PARTITION_FILES
    }
    fit_urls = frames["fit"]["url"]
    validation_urls = frames["validation"]["url"]
    if not set(fit_urls).isdisjoint(set(validation_urls)):
        raise SharedPartitionError("Fit and Validation partitions contain overlapping URLs.")

    return SharedPartitions(
        x_fit=fit_urls,
        y_fit=frames["fit"]["label"],
        x_validation=validation_urls,
        y_validation=frames["validation"]["label"],
        fit_sha256=raw_hashes["fit"],
        validation_sha256=raw_hashes["validation"],
        partition_manifest_sha256=_canonical_manifest_sha256(manifest),
        feature_extraction_sha256=_normalized_feature_extraction_sha256(),
        manifest=manifest,
    )


def add_training_args(parser: argparse.ArgumentParser, default_output: str):
    parser.add_argument(
        "--partitions-dir",
        type=Path,
        default=DEFAULT_PARTITIONS_DIR,
        help="Shared Fit/Validation partition directory (default: %(default)s)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Model seed; does not split data")
    parser.add_argument("--output", default=default_output)
    return parser


def _domain_split_all(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
    validation_size: float,
):
    """Return Fit, Validation, and Test from the established domain split."""
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
    return (
        x_fit.reset_index(drop=True),
        x_validation.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_fit.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        y_test.reset_index(drop=True),
        summary,
    )


def domain_split(data: pd.DataFrame, test_size: float, random_state: int, validation_size: float):
    """Return Fit and Validation while retaining Test, as before."""
    (
        x_fit,
        x_validation,
        _x_test,
        y_fit,
        y_validation,
        _y_test,
        summary,
    ) = _domain_split_all(data, test_size, random_state, validation_size)
    # Test remains intentionally unavailable to existing training callers.
    return x_fit, x_validation, y_fit, y_validation, summary


def train_and_save(
    model,
    model_name: str,
    args,
    *,
    timing_sample_size: int | None = None,
) -> dict:
    """Fit only on shared Fit, score shared Validation, and save a v2 artifact."""
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model artifact: {output}")
    partitions = load_shared_partitions(args.partitions_dir)

    start = time.perf_counter()
    model.fit(partitions.x_fit.tolist(), partitions.y_fit)
    fit_seconds = time.perf_counter() - start
    metrics = evaluate_classifier(
        model,
        partitions.x_validation.tolist(),
        partitions.y_validation,
        f"{model_name}_validation",
        fit_time_seconds=fit_seconds,
        save_plot=False,
    )

    prediction_timing = None
    if timing_sample_size is not None:
        if timing_sample_size < 1:
            raise ValueError("timing_sample_size must be positive.")
        sample = partitions.x_validation.iloc[:timing_sample_size].tolist()
        start = time.perf_counter()
        model.predict(sample)
        elapsed = time.perf_counter() - start
        prediction_timing = {
            "sample_size": len(sample),
            "elapsed_seconds": elapsed,
            "ms_per_url": elapsed * 1000 / max(len(sample), 1),
            "measurement": "end-to-end predict, including in-pipeline URL feature extraction",
            "source": "first rows of shared Validation in original order",
        }

    split_info = {
        "fit_rows": len(partitions.x_fit),
        "validation_rows": len(partitions.x_validation),
        "inner_random_state": partitions.manifest["split"]["inner_random_state"],
        "strategy": partitions.manifest["split"]["strategy"],
    }
    training_config = {
        "fit_partition": "shared fit.csv",
        "validation_partition": "shared validation.csv",
    }
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None:
        parameters = classifier.get_params(deep=False)
        if "random_state" in parameters:
            training_config["random_state"] = args.random_state
        if "n_neighbors" in parameters:
            training_config["neighbors"] = parameters["n_neighbors"]
    artifact = {
        "artifact_version": 2,
        "model": model,
        "model_type": "raw_url_features",
        "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_extraction_sha256": partitions.feature_extraction_sha256,
        "label_mapping": {
            "source_to_internal": dict(PHIUSIIL_LABEL_MAPPING),
            "internal": {str(key): value for key, value in LABEL_NAMES.items()},
        },
        "training_config": training_config,
        "partition": {
            "fit_sha256": partitions.fit_sha256,
            "validation_sha256": partitions.validation_sha256,
            "partition_manifest_sha256": partitions.partition_manifest_sha256,
            "fit_rows": len(partitions.x_fit),
            "validation_rows": len(partitions.x_validation),
        },
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
    if prediction_timing is not None:
        artifact["prediction_timing"] = prediction_timing

    output.parent.mkdir(parents=True, exist_ok=True)
    created_identity = None
    try:
        with output.open("xb") as stream:
            file_stat = os.fstat(stream.fileno())
            created_identity = (file_stat.st_dev, file_stat.st_ino)
            joblib.dump(artifact, stream)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = joblib.load(output)
        probe = loaded["model"].predict(partitions.x_validation.iloc[:1].tolist())
        if len(probe) != 1:
            raise RuntimeError("Reloaded artifact failed its prediction check.")
    except BaseException:
        if created_identity is not None:
            try:
                current = output.stat()
                if (current.st_dev, current.st_ino) == created_identity:
                    output.unlink()
            except OSError:
                pass
        raise

    result = {
        "model": model_name,
        "training_time_seconds": fit_seconds,
        "validation_metrics": metrics,
        "split": split_info,
        "partition": artifact["partition"],
        "artifact": str(output),
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "artifact_bytes": output.stat().st_size,
    }
    if prediction_timing is not None:
        result["prediction_timing"] = prediction_timing
    return result


def write_comparison_report(report: dict, path: str = "reports/model_comparison.json"):
    """Write a new comparison report without overwriting existing results."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
