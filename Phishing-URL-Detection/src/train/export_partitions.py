"""Export the fixed, registered-domain-isolated project partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import tldextract

from src.feature_extraction import FEATURE_COLUMNS
from src.preprocessing import LABEL_NAMES, PHIUSIIL_LABEL_MAPPING, registered_domain, load_dataset
from src.train._common import _domain_split_all


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_SHARED_PATH = PROJECT_ROOT / "data" / "partitions" / "shared"
DEFAULT_HOLDOUT_PATH = PROJECT_ROOT / "data" / "partitions" / "holdout"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "reports" / "baseline_metrics.json"
EXPECTED_DATASET_SHA256 = "A236549CD369CD80BD478FF8E1779CBF44C58D5C3F79F7A51A1ADBED7D06D1C6"

OUTER_TEST_SIZE = 0.20
OUTER_RANDOM_STATE = 42
INNER_VALIDATION_SIZE = 0.15
INNER_RANDOM_STATE = 43
CANDIDATE_ATTEMPTS = 30
BASELINE_FIELDS = (
    "fit_rows",
    "validation_rows",
    "test_rows",
    "fit_domains",
    "validation_domains",
    "test_domains",
)


class PartitionExportError(RuntimeError):
    """Raised when a safe, verifiable partition export cannot be completed."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_same_or_nested(path: Path, other: Path) -> bool:
    """Return whether path is equal to or below other."""
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _resolve_and_check_outputs(shared_value: str | Path, holdout_value: str | Path):
    shared = Path(shared_value).expanduser().resolve()
    holdout = Path(holdout_value).expanduser().resolve()
    if _is_same_or_nested(shared, holdout) or _is_same_or_nested(holdout, shared):
        raise PartitionExportError(
            "Shared and holdout output paths must be separate and may not contain one another."
        )

    destinations = {
        "fit": shared / "fit.csv",
        "validation": shared / "validation.csv",
        "manifest": shared / "split_manifest.json",
        "info": shared / "partition_info.txt",
        "test": holdout / "test.csv",
    }
    present = [str(path) for path in destinations.values() if path.exists()]
    if present:
        raise PartitionExportError(
            "Refusing to overwrite existing output file(s): " + ", ".join(present)
        )
    return shared, holdout, destinations


def _read_baseline(path: str | Path) -> dict:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PartitionExportError(f"Historical baseline file does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartitionExportError(f"Could not read a valid historical baseline: {path}") from exc

    if not isinstance(document, dict) or not isinstance(document.get("split"), dict):
        raise PartitionExportError("Historical baseline must contain a split object.")
    split = document["split"]
    missing = [field for field in BASELINE_FIELDS if field not in split]
    if missing:
        raise PartitionExportError(
            "Historical baseline split is missing required field(s): " + ", ".join(missing)
        )
    for field in BASELINE_FIELDS:
        value = split[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PartitionExportError(
                f"Historical baseline split field {field!r} must be a nonnegative integer."
            )
    return {field: split[field] for field in BASELINE_FIELDS}


def _audit_metadata(data: pd.DataFrame) -> dict:
    audit = data.attrs.get("audit")
    if not isinstance(audit, dict) or "rows_input" not in audit:
        raise PartitionExportError("load_dataset() did not provide valid audit metadata with rows_input.")
    rows_input = audit["rows_input"]
    if isinstance(rows_input, bool) or not isinstance(rows_input, int) or rows_input < 0:
        raise PartitionExportError("Dataset audit rows_input must be a nonnegative integer.")
    return audit


def _label_distribution(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().to_dict()
    return {LABEL_NAMES[int(label)]: int(counts.get(label, 0)) for label in sorted(LABEL_NAMES)}


def _partition_domains(urls: pd.Series) -> set[str]:
    return {registered_domain(str(url)) for url in urls}


def _csv_bytes(urls: pd.Series, labels: pd.Series) -> bytes:
    frame = pd.DataFrame({"url": urls.tolist(), "label": labels.astype(int).tolist()})
    output = io.StringIO(newline="")
    frame.to_csv(output, index=False, columns=["url", "label"], lineterminator="\n")
    return output.getvalue().encode("utf-8")


def _git_metadata() -> dict:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PartitionExportError("Could not capture read-only Git metadata for the manifest.") from exc
    return {
        "commit": commit,
        "working_tree_dirty": bool(status.strip()),
        "captured_at_export_time": True,
    }


def _code_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "src" / "preprocessing.py",
        PROJECT_ROOT / "src" / "train" / "_common.py",
        PROJECT_ROOT / "src" / "train" / "export_partitions.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT).as_posix()): _sha256_file(path)
        for path in paths
    }


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise PartitionExportError(f"Required package metadata is unavailable: {distribution}") from exc


def _environment_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "pandas": _package_version("pandas"),
        "scikit_learn": _package_version("scikit-learn"),
        "tldextract": _package_version("tldextract"),
        "tldextract_suffix_list_urls": [],
    }


def _make_manifest(
    *,
    source_path: Path,
    source_sha256: str,
    source_columns: list[str],
    audit: dict,
    processed_rows: int,
    partitions: dict,
    split_summary: dict,
    overlap: dict,
    baseline_comparison: dict,
    shared_path: Path,
    test_csv_path: Path,
    code_hashes: dict,
) -> dict:
    actual_ratios = {
        name: rows / processed_rows
        for name, rows in (
            ("fit", split_summary["fit_rows"]),
            ("validation", split_summary["validation_rows"]),
            ("test", split_summary["test_rows"]),
        )
    }
    return {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "source_filename": source_path.name,
            "source_sha256": source_sha256,
            "rows_input": audit["rows_input"],
            "processed_rows": processed_rows,
            "source_columns": source_columns,
            "audit": audit,
        },
        "label_mapping": {
            "source_to_internal": dict(PHIUSIIL_LABEL_MAPPING),
            "internal": {str(label): name for label, name in LABEL_NAMES.items()},
        },
        "split": {
            "strategy": "GroupShuffleSplit by registered domain",
            "outer_test_size": OUTER_TEST_SIZE,
            "outer_random_state": OUTER_RANDOM_STATE,
            "inner_validation_size": INNER_VALIDATION_SIZE,
            "inner_random_state": INNER_RANDOM_STATE,
            "candidate_attempts": CANDIDATE_ATTEMPTS,
            "actual_ratios": actual_ratios,
            "registered_domain_overlap": overlap,
            "baseline_comparison": baseline_comparison,
            "historical_identity_limit": (
                "Matching the six historical summary counts does not prove exact partition identity; "
                "historical partition hashes are unavailable."
            ),
        },
        "partitions": {
            "fit": {
                "path": "fit.csv",
                "rows": split_summary["fit_rows"],
                "sha256": partitions["fit"]["sha256"],
                "label_distribution": partitions["fit"]["label_distribution"],
                "registered_domain_count": split_summary["fit_domains"],
                "location": "shared",
            },
            "validation": {
                "path": "validation.csv",
                "rows": split_summary["validation_rows"],
                "sha256": partitions["validation"]["sha256"],
                "label_distribution": partitions["validation"]["label_distribution"],
                "registered_domain_count": split_summary["validation_domains"],
                "location": "shared",
            },
            "test": {
                "path": os.path.relpath(test_csv_path, shared_path),
                "rows": split_summary["test_rows"],
                "sha256": partitions["test"]["sha256"],
                "location": "holdout (not shared)",
            },
        },
        "features": {"count": len(FEATURE_COLUMNS), "names": list(FEATURE_COLUMNS)},
        "environment": _environment_metadata(),
        "git": _git_metadata(),
        "code_sha256": code_hashes,
        "test_usage_policy": {
            "fit": "Model training",
            "validation": "Model selection and tuning",
            "test": "Final evaluation only; never for training, tuning, or model selection",
        },
    }


def _partition_info(manifest: dict) -> bytes:
    dataset = manifest["dataset"]
    split = manifest["split"]
    parts = manifest["partitions"]
    lines = [
        "Shared Phishing URL Detection partitions",
        f"Dataset filename: {dataset['source_filename']}",
        f"Dataset SHA-256: {dataset['source_sha256']}",
        f"Original input rows: {dataset['rows_input']}",
        f"Processed rows: {dataset['processed_rows']}",
        f"Fit rows: {parts['fit']['rows']}",
        f"Validation rows: {parts['validation']['rows']}",
        f"Fit label distribution: {json.dumps(parts['fit']['label_distribution'], sort_keys=True)}",
        f"Validation label distribution: {json.dumps(parts['validation']['label_distribution'], sort_keys=True)}",
        f"Fit registered-domain count: {parts['fit']['registered_domain_count']}",
        f"Validation registered-domain count: {parts['validation']['registered_domain_count']}",
        f"Registered-domain overlaps: {json.dumps(split['registered_domain_overlap'], sort_keys=True)}",
        (
            "Split parameters: "
            f"outer_test_size={split['outer_test_size']}, outer_random_state={split['outer_random_state']}, "
            f"inner_validation_size={split['inner_validation_size']}, "
            f"inner_random_state={split['inner_random_state']}, "
            f"candidate_attempts={split['candidate_attempts']}, strategy={split['strategy']}"
        ),
        f"Fit SHA-256: {parts['fit']['sha256']}",
        f"Validation SHA-256: {parts['validation']['sha256']}",
        f"Test rows: {parts['test']['rows']}",
        f"Test SHA-256: {parts['test']['sha256']}",
        f"Test holdout path: {parts['test']['path']}",
        "Historical baseline comparison:",
    ]
    for name, result in split["baseline_comparison"].items():
        if name == "test_domains":
            lines.append(f"  {name}: match={str(result['match']).lower()}")
        else:
            lines.append(
                f"  {name}: expected={result['expected']}, actual={result['actual']}, "
                f"match={str(result['match']).lower()}"
            )
    lines.extend(
        [
            "Baseline limitation: matching counts do not prove exact historical partition identity; historical partition hashes are unavailable.",
            "TEST HOLDOUT WARNING: do not share Test or use it for training, tuning, or model selection. Reserve it for final evaluation.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _ensure_directory(path: Path, created_directories: list[tuple[Path, int, int]]) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise PartitionExportError(f"Could not find an existing parent for output directory: {path}")
        cursor = parent
    if not cursor.is_dir():
        raise PartitionExportError(f"Output parent is not a directory: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise PartitionExportError(f"Output path is not a directory: {directory}")
        else:
            stat = directory.stat()
            created_directories.append((directory, stat.st_dev, stat.st_ino))


def _write_new_file(
    path: Path,
    content: bytes,
    created_files: list[tuple[Path, int, int]],
) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    stat = os.fstat(descriptor)
    created_files.append((path, stat.st_dev, stat.st_ino))
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _rollback(
    created_files: list[tuple[Path, int, int]],
    created_directories: list[tuple[Path, int, int]],
) -> None:
    for path, device, inode in reversed(created_files):
        try:
            current = path.stat()
            if (current.st_dev, current.st_ino) == (device, inode):
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    for path, device, inode in reversed(created_directories):
        try:
            current = path.stat()
            if (current.st_dev, current.st_ino) == (device, inode) and path.is_dir():
                path.rmdir()
        except OSError:
            pass


def main(
    argv: list[str] | None = None,
    *,
    expected_sha256: str = EXPECTED_DATASET_SHA256,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default=str(DEFAULT_SHARED_PATH))
    parser.add_argument("--holdout-output", default=str(DEFAULT_HOLDOUT_PATH))
    args = parser.parse_args(argv)

    shared_path, holdout_path, output_paths = _resolve_and_check_outputs(
        args.output, args.holdout_output
    )
    data_path = Path(args.data).expanduser().resolve()
    try:
        source_sha256 = _sha256_file(data_path)
    except OSError as exc:
        raise PartitionExportError(f"Could not read dataset for SHA-256 verification: {data_path}") from exc
    if source_sha256.lower() != expected_sha256.lower():
        raise PartitionExportError(
            "Dataset SHA-256 does not match the approved source. "
            f"Expected {expected_sha256}; received {source_sha256}."
        )

    try:
        data = load_dataset(str(data_path))
    except Exception as exc:
        raise PartitionExportError(f"Could not preprocess the source dataset: {exc}") from exc
    audit = _audit_metadata(data)
    if data.empty:
        raise PartitionExportError("Preprocessed dataset is empty.")
    try:
        source_columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    except Exception as exc:
        raise PartitionExportError("Could not read original dataset column names.") from exc

    baseline = _read_baseline(baseline_path)
    (
        x_fit,
        x_validation,
        x_test,
        y_fit,
        y_validation,
        y_test,
        summary,
    ) = _domain_split_all(data, OUTER_TEST_SIZE, OUTER_RANDOM_STATE, INNER_VALIDATION_SIZE)

    partitions_raw = {
        "fit": (x_fit, y_fit),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }
    row_counts = {name: len(urls) for name, (urls, _) in partitions_raw.items()}
    if any(count <= 0 for count in row_counts.values()):
        raise PartitionExportError("Fit, Validation, and Test must all contain rows.")
    if sum(row_counts.values()) != len(data):
        raise PartitionExportError("Partition row counts do not sum to the processed dataset row count.")

    domain_sets = {
        name: _partition_domains(urls)
        for name, (urls, _) in partitions_raw.items()
    }
    overlap = {
        "fit_validation": len(domain_sets["fit"] & domain_sets["validation"]),
        "fit_test": len(domain_sets["fit"] & domain_sets["test"]),
        "validation_test": len(domain_sets["validation"] & domain_sets["test"]),
    }
    if any(overlap.values()):
        raise PartitionExportError(f"Registered-domain overlap detected: {overlap}")
    expected_split_summary = {
        "fit_rows": row_counts["fit"],
        "validation_rows": row_counts["validation"],
        "test_rows": row_counts["test"],
        "fit_domains": len(domain_sets["fit"]),
        "validation_domains": len(domain_sets["validation"]),
        "test_domains": len(domain_sets["test"]),
    }
    if any(summary.get(key) != value for key, value in expected_split_summary.items()):
        raise PartitionExportError("Split helper summary does not match the partition data.")
    baseline_comparison = {
        field: {
            "expected": baseline[field],
            "actual": expected_split_summary[field],
            "match": baseline[field] == expected_split_summary[field],
        }
        for field in BASELINE_FIELDS
    }
    if not all(item["match"] for item in baseline_comparison.values()):
        mismatches = [name for name, item in baseline_comparison.items() if not item["match"]]
        raise PartitionExportError(
            "Current partition summary does not match the historical baseline: "
            + ", ".join(mismatches)
        )

    csv_contents = {
        name: _csv_bytes(urls, labels)
        for name, (urls, labels) in partitions_raw.items()
    }
    partition_metadata = {
        name: {
            "sha256": _sha256_bytes(content),
            **({"label_distribution": _label_distribution(labels)} if name != "test" else {}),
        }
        for (name, (_, labels)), content in zip(partitions_raw.items(), csv_contents.values())
    }
    comparison_for_shared = {
        name: ({"match": item["match"]} if name == "test_domains" else item)
        for name, item in baseline_comparison.items()
    }
    safe_summary = {**summary, **expected_split_summary}
    manifest = _make_manifest(
        source_path=data_path,
        source_sha256=source_sha256,
        source_columns=source_columns,
        audit=audit,
        processed_rows=len(data),
        partitions=partition_metadata,
        split_summary=safe_summary,
        overlap=overlap,
        baseline_comparison=comparison_for_shared,
        shared_path=shared_path,
        test_csv_path=output_paths["test"],
        code_hashes=_code_hashes(),
    )
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    info_bytes = _partition_info(manifest)

    created_files: list[tuple[Path, int, int]] = []
    created_directories: list[tuple[Path, int, int]] = []
    try:
        _ensure_directory(shared_path, created_directories)
        _ensure_directory(holdout_path, created_directories)
        _write_new_file(output_paths["fit"], csv_contents["fit"], created_files)
        _write_new_file(output_paths["validation"], csv_contents["validation"], created_files)
        _write_new_file(output_paths["test"], csv_contents["test"], created_files)
        _write_new_file(output_paths["manifest"], manifest_bytes, created_files)
        _write_new_file(output_paths["info"], info_bytes, created_files)

        written_hashes = {
            name: _sha256_file(output_paths[name])
            for name in ("fit", "validation", "test")
        }
        for name in written_hashes:
            if written_hashes[name] != partition_metadata[name]["sha256"]:
                raise PartitionExportError(f"Written {name} CSV hash does not match its manifest value.")
        if _sha256_file(data_path) != source_sha256:
            raise PartitionExportError("Source dataset SHA-256 changed during export; outputs were rolled back.")
    except BaseException:
        _rollback(created_files, created_directories)
        raise

    return manifest


if __name__ == "__main__":
    main()
