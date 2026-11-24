"""Synthetic-only tests for the shared Fit/Validation training workflow."""
from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_extraction import FEATURE_COLUMNS
from src.preprocessing import LABEL_NAMES, PHIUSIIL_LABEL_MAPPING
from src.train import compare_models
from src.train import train_knn, train_logistic_regression, train_random_forest
from src.train._common import (
    DEFAULT_PARTITIONS_DIR,
    PROJECT_ROOT,
    SharedPartitionError,
    _canonical_manifest_sha256,
    _normalized_feature_extraction_sha256,
    _safe_partition_filename,
    load_shared_partitions,
)


TRAINERS = (
    ("random_forest", train_random_forest, "random_forest_candidate.joblib"),
    ("logistic_regression", train_logistic_regression, "logistic_regression.joblib"),
    ("knn", train_knn, "knn.joblib"),
)
REMOVED_TRAINER_OPTIONS = (
    "--data",
    "--url-col",
    "--label-col",
    "--test-size",
    "--validation-size",
    "--split-strategy",
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _synthetic_rows(prefix: str, count: int):
    rows = []
    for index in range(count):
        label = index % 2
        scheme = "https" if index % 3 else "http"
        path = ("/login" if index % 3 == 0 else "/account/reset" if index % 3 == 1 else f"/offer/{index}")
        url = f"{scheme}://{prefix}{index:04d}.example.{ 'com' if label == 0 else 'net' }{path}"
        rows.append((url, label))
    return rows


def _csv_bytes(rows) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["url", "label"])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _write_partition_set(directory: Path, *, test_meta: dict | None = None):
    directory.mkdir(parents=True)
    fit_rows = _synthetic_rows("fit", 100)
    validation_rows = _synthetic_rows("val", 32)
    fit_content = _csv_bytes(fit_rows)
    validation_content = _csv_bytes(validation_rows)
    (directory / "fit.csv").write_bytes(fit_content)
    (directory / "validation.csv").write_bytes(validation_content)

    partitions = {
        "fit": {"path": "fit.csv", "rows": len(fit_rows), "sha256": _sha(fit_content), "location": "shared"},
        "validation": {
            "path": "validation.csv",
            "rows": len(validation_rows),
            "sha256": _sha(validation_content),
            "location": "shared",
        },
    }
    if test_meta is not None:
        partitions["test"] = test_meta
    manifest = {
        "manifest_version": 1,
        "created_utc": "synthetic-time",
        "dataset": {"source_filename": "synthetic.csv", "source_sha256": "a" * 64},
        "label_mapping": {
            "source_to_internal": dict(PHIUSIIL_LABEL_MAPPING),
            "internal": {str(key): value for key, value in LABEL_NAMES.items()},
        },
        "split": {
            "strategy": "GroupShuffleSplit by registered domain",
            "inner_random_state": 43,
        },
        "partitions": partitions,
        "features": {"count": len(FEATURE_COLUMNS), "names": list(FEATURE_COLUMNS)},
    }
    (directory / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


@pytest.fixture()
def partition_dir(tmp_path):
    return tmp_path / "partitions" / "shared"


def _install_test_file_access_block(monkeypatch):
    """Block any filesystem or CSV API that is handed a test.csv pathname."""
    real_open = builtins.open
    real_read_csv = pd.read_csv
    real_stat = os.stat
    real_os_exists = os.path.exists
    real_os_isfile = os.path.isfile
    real_path_exists = Path.exists
    real_path_stat = Path.stat
    real_path_open = Path.open
    real_path_read_bytes = Path.read_bytes
    real_path_read_text = Path.read_text

    def is_test_path(value) -> bool:
        try:
            text = os.fspath(value)
        except TypeError:
            return False
        if not isinstance(text, (str, bytes)):
            return False
        if isinstance(text, bytes):
            text = os.fsdecode(text)
        normalized = text.replace("/", "\\")
        basename = normalized.rsplit("\\", 1)[-1]
        return basename.casefold() == "test.csv"

    def reject(value):
        if is_test_path(value):
            raise AssertionError("Training/compare attempted filesystem access to the holdout Test file.")

    def guarded_open(file, *args, **kwargs):
        reject(file)
        return real_open(file, *args, **kwargs)

    def guarded_read_csv(filepath_or_buffer, *args, **kwargs):
        reject(filepath_or_buffer)
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        reject(path)
        return real_stat(path, *args, **kwargs)

    def guarded_os_exists(path):
        reject(path)
        return real_os_exists(path)

    def guarded_os_isfile(path):
        reject(path)
        return real_os_isfile(path)

    def guarded_path_exists(path):
        reject(path)
        return real_path_exists(path)

    def guarded_path_stat(path, *args, **kwargs):
        reject(path)
        return real_path_stat(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        reject(path)
        return real_path_open(path, *args, **kwargs)

    def guarded_path_read_bytes(path):
        reject(path)
        return real_path_read_bytes(path)

    def guarded_path_read_text(path, *args, **kwargs):
        reject(path)
        return real_path_read_text(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(os.path, "exists", guarded_os_exists)
    monkeypatch.setattr(os.path, "isfile", guarded_os_isfile)
    monkeypatch.setattr(Path, "exists", guarded_path_exists)
    monkeypatch.setattr(Path, "stat", guarded_path_stat)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_path_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_path_read_text)
    return reject


def _run_all_trainers_and_compare(directory: Path, artifact_root: Path):
    artifact_root.mkdir(parents=True)
    outputs = {}
    for name, module, filename in TRAINERS:
        output = artifact_root / filename
        args = ["--partitions-dir", str(directory), "--output", str(output), "--random-state", "17"]
        if name == "knn":
            args += ["--neighbors", "3", "--timing-sample-size", "7"]
        module.main(args)
        outputs[name] = output

    report_path = artifact_root / "comparison.json"
    report = compare_models.main(
        [
            "--partitions-dir", str(directory),
            "--rf-artifact", str(outputs["random_forest"]),
            "--logistic-artifact", str(outputs["logistic_regression"]),
            "--knn-artifact", str(outputs["knn"]),
            "--output-report", str(report_path),
        ]
    )
    return outputs, report_path, report


@pytest.mark.parametrize("poison_name", [None, "test.csv", "Test.csv", "TEST.CSV"])
def test_three_trainers_and_comparison_never_touch_holdout(
    partition_dir, tmp_path, monkeypatch, capsys, poison_name
):
    manifest = _write_partition_set(
        partition_dir,
        test_meta={"path": "../holdout/test.csv", "rows": 1, "sha256": "b" * 64, "location": "holdout (not shared)"},
    )
    if poison_name is not None:
        holdout = partition_dir.parent / "holdout"
        holdout.mkdir()
        (holdout / poison_name).write_bytes(b"POISON HOLDOUT PAYLOAD -- MUST NOT BE OPENED")

    reject_test_path = _install_test_file_access_block(monkeypatch)
    artifacts, report_path, report = _run_all_trainers_and_compare(
        partition_dir, tmp_path / "artifacts"
    )
    capsys.readouterr()

    assert set(artifacts) == {"random_forest", "logistic_regression", "knn"}
    assert report_path.is_file()
    assert report["validation_rows"] == 32
    assert set(report["validation_metrics"]) == {"random_forest", "logistic_regression", "knn"}
    assert "test" not in json.dumps(report).casefold()
    assert "test" in manifest["partitions"]

    # Exercise the guard against all requested APIs with relative and absolute paths
    # and case variants. Rejection occurs before any real filesystem operation.
    relative_paths = ("holdout/test.csv", "holdout/Test.csv", "holdout/TEST.CSV")
    absolute_paths = tuple(str(tmp_path / "holdout" / name) for name in ("test.csv", "Test.csv", "TEST.CSV"))
    for value in (*relative_paths, *absolute_paths):
        actions = (
            lambda v=value: builtins.open(v, "rb"),
            lambda v=value: pd.read_csv(v),
            lambda v=value: os.stat(v),
            lambda v=value: os.path.exists(v),
            lambda v=value: os.path.isfile(v),
            lambda v=value: Path(v).exists(),
            lambda v=value: Path(v).stat(),
            lambda v=value: Path(v).open("rb"),
            lambda v=value: Path(v).read_bytes(),
            lambda v=value: Path(v).read_text(encoding="utf-8"),
        )
        for action in actions:
            with pytest.raises(AssertionError, match="holdout Test"):
                action()
    with pytest.raises(AssertionError, match="holdout Test"):
        reject_test_path("relative/path/TEST.CSV")


def test_loader_reads_fit_and_validation_with_canonical_hash_and_feature_hash(partition_dir):
    manifest = _write_partition_set(partition_dir, test_meta={"path": "../holdout/test.csv"})
    # CRLF changes in the JSON file must not alter its canonical manifest hash.
    manifest_path = partition_dir / "split_manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"\n", b"\r\n"))
    loaded = load_shared_partitions(partition_dir)

    assert len(loaded.x_fit) == loaded.y_fit.size == 100
    assert len(loaded.x_validation) == loaded.y_validation.size == 32
    assert loaded.partition_manifest_sha256 == _canonical_manifest_sha256(manifest)
    assert loaded.feature_extraction_sha256 == _normalized_feature_extraction_sha256()
    source = (PROJECT_ROOT / "src" / "feature_extraction.py").read_bytes()
    normalized = source.replace(b"\r\n", b"\n")
    assert loaded.feature_extraction_sha256 == hashlib.sha256(normalized).hexdigest()
    assert loaded.x_fit.iloc[0].startswith("http://fit")
    assert loaded.x_validation.iloc[0].startswith("http://val")


@pytest.mark.parametrize(
    "unsafe",
    [
        "../fit.csv", "..\\fit.csv", "sub/fit.csv", "C:\\fit.csv", "/fit.csv",
        "CON.csv", "PRN.csv", "AUX.csv", "NUL.csv", "COM1.csv", "COM9.csv",
        "LPT1.csv", "LPT9.csv", "FIT.CSV",
    ],
)
def test_loader_rejects_unsafe_manifest_partition_paths(partition_dir, unsafe):
    manifest = _write_partition_set(partition_dir)
    manifest["partitions"]["fit"]["path"] = unsafe
    (partition_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SharedPartitionError):
        load_shared_partitions(partition_dir)


def test_loader_reports_expected_and_actual_hash_on_mismatch(partition_dir):
    manifest = _write_partition_set(partition_dir)
    path = partition_dir / "fit.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SharedPartitionError, match="SHA-256 mismatch") as error:
        load_shared_partitions(partition_dir)
    assert "Expected:" in str(error.value) and "Actual:" in str(error.value)


@pytest.mark.parametrize(
    "replacement",
    [
        b"url,other\nhttps://host.example.com/,0\n",
        b"url,label,extra\nhttps://host.example.com/,0,x\n",
        b"url,label\nhttps://host.example.com/,1.0\n",
        b"url,label\nhttps://host.example.com/,01\n",
        b"url,label\nhttps://host.example.com/,+1\n",
        b"url,label\nhttps://host.example.com/,2\n",
        b"url,label\n,0\n",
    ],
)
def test_loader_rejects_bad_schema_labels_and_empty_urls(partition_dir, replacement):
    manifest = _write_partition_set(partition_dir)
    validation_path = partition_dir / "validation.csv"
    validation_path.write_bytes(replacement)
    manifest["partitions"]["validation"]["rows"] = 1
    manifest["partitions"]["validation"]["sha256"] = _sha(replacement)
    (partition_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SharedPartitionError):
        load_shared_partitions(partition_dir)


def test_loader_rejects_row_count_mismatch_and_fit_validation_url_overlap(partition_dir):
    manifest = _write_partition_set(partition_dir)
    manifest["partitions"]["fit"]["rows"] += 1
    (partition_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SharedPartitionError, match="row count mismatch"):
        load_shared_partitions(partition_dir)

    fit_rows = _synthetic_rows("fit", 100)
    validation_rows = _synthetic_rows("val", 32)
    validation_rows[0] = fit_rows[0]
    fit_content, validation_content = _csv_bytes(fit_rows), _csv_bytes(validation_rows)
    (partition_dir / "fit.csv").write_bytes(fit_content)
    (partition_dir / "validation.csv").write_bytes(validation_content)
    manifest["partitions"]["fit"].update(rows=len(fit_rows), sha256=_sha(fit_content))
    manifest["partitions"]["validation"].update(rows=len(validation_rows), sha256=_sha(validation_content))
    (partition_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SharedPartitionError, match="overlapping URLs"):
        load_shared_partitions(partition_dir)


def test_loader_rejects_missing_manifest_sections_and_metadata(partition_dir):
    _write_partition_set(partition_dir)
    manifest_path = partition_dir / "split_manifest.json"
    for content in (
        {},
        {"manifest_version": 1},
        {"manifest_version": 1, "partitions": {"validation": {}}},
    ):
        manifest_path.write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(SharedPartitionError):
            load_shared_partitions(partition_dir)


def test_cli_removes_split_options_and_keeps_shared_partition_option(capsys, tmp_path):
    for _name, module, _filename in TRAINERS:
        parser = module.build_parser()
        for option in REMOVED_TRAINER_OPTIONS:
            with pytest.raises(SystemExit) as error:
                parser.parse_args([option, "unused"])
            assert error.value.code == 2
            capsys.readouterr()
        with pytest.raises(SystemExit) as help_result:
            parser.parse_args(["--help"])
        assert help_result.value.code == 0
        help_text = capsys.readouterr().out
        assert "".join(str(DEFAULT_PARTITIONS_DIR).split()) in "".join(help_text.split())
        parsed = parser.parse_args(["--partitions-dir", str(tmp_path), "--output", str(tmp_path / "m.joblib")])
        assert parsed.partitions_dir == tmp_path

    parser = compare_models.build_parser()
    for option in (*REMOVED_TRAINER_OPTIONS, "--baseline-report"):
        with pytest.raises(SystemExit) as error:
            parser.parse_args([option, "unused"])
        assert error.value.code == 2
        capsys.readouterr()
    with pytest.raises(SystemExit) as help_result:
        parser.parse_args(["--help"])
    assert help_result.value.code == 0
    assert "".join(str(DEFAULT_PARTITIONS_DIR).split()) in "".join(capsys.readouterr().out.split())


def test_cli_help_works_from_another_working_directory(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib-cache")
    modules = [module.__name__ for _, module, _ in TRAINERS] + [compare_models.__name__]
    for module_name in modules:
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "".join(str(DEFAULT_PARTITIONS_DIR).split()) in "".join(result.stdout.split())


def test_trained_artifacts_are_v2_loadable_and_do_not_contain_test_metadata(
    partition_dir, tmp_path, monkeypatch, capsys
):
    _write_partition_set(partition_dir)
    artifacts, _report_path, _report = _run_all_trainers_and_compare(
        partition_dir, tmp_path / "artifacts"
    )
    capsys.readouterr()
    from src.predict import predict_url

    for name, path in artifacts.items():
        artifact = joblib.load(path)
        assert artifact["artifact_version"] == 2
        assert artifact["model_name"] == name
        assert artifact["feature_columns"] == FEATURE_COLUMNS
        assert len(artifact["feature_extraction_sha256"]) == 64
        assert set(artifact["partition"]) == {
            "fit_sha256", "validation_sha256", "partition_manifest_sha256",
            "fit_rows", "validation_rows",
        }
        assert set(artifact["split"]) == {
            "fit_rows", "validation_rows", "inner_random_state", "strategy",
        }
        assert not any("test" in key.casefold() for key in artifact if isinstance(key, str))
        assert artifact["partition"]["fit_rows"] == 100
        assert artifact["partition"]["validation_rows"] == 32
        prediction = predict_url("https://predict.example.org/check", str(path))
        assert prediction["prediction"] in (0, 1)
    assert artifacts["knn"]
    knn = joblib.load(artifacts["knn"])
    assert knn["prediction_timing"]["sample_size"] == 7
    assert "feature extraction" in knn["prediction_timing"]["measurement"]


def test_training_refuses_to_overwrite_existing_artifact(partition_dir, tmp_path):
    _write_partition_set(partition_dir)
    output = tmp_path / "already-there.joblib"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        train_random_forest.main(
            ["--partitions-dir", str(partition_dir), "--output", str(output)]
        )
    assert output.read_bytes() == b"preserve"


def test_compare_rejects_incompatible_partition_or_feature_metadata(partition_dir, tmp_path):
    _write_partition_set(partition_dir)
    loaded = load_shared_partitions(partition_dir)
    good = {
        "artifact_version": 2,
        "model_name": "random_forest",
        "model": object(),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_extraction_sha256": loaded.feature_extraction_sha256,
        "partition": {
            "fit_sha256": loaded.fit_sha256,
            "validation_sha256": loaded.validation_sha256,
            "partition_manifest_sha256": loaded.partition_manifest_sha256,
            "fit_rows": len(loaded.x_fit),
            "validation_rows": len(loaded.x_validation),
        },
    }
    path = tmp_path / "rf.joblib"
    output_report = tmp_path / "comparison.json"

    cases = (
        ("partition", "fit_sha256", "0" * 64),
        ("partition", "partition_manifest_sha256", "1" * 64),
        ("top", "feature_extraction_sha256", "2" * 64),
        ("top", "feature_columns", list(reversed(FEATURE_COLUMNS))),
    )
    for location, field, wrong in cases:
        candidate = dict(good)
        candidate["partition"] = dict(good["partition"])
        if location == "partition":
            candidate["partition"][field] = wrong
        else:
            candidate[field] = wrong
        joblib.dump(candidate, path)
        with pytest.raises(ValueError, match="Expected:.*actual"):
            compare_models.main(
                [
                    "--partitions-dir", str(partition_dir),
                    "--rf-artifact", str(path),
                    "--logistic-artifact", str(tmp_path / "missing-lr.joblib"),
                    "--knn-artifact", str(tmp_path / "missing-knn.joblib"),
                    "--output-report", str(output_report),
                ]
            )
        assert not output_report.exists()


def test_compare_does_not_load_legacy_model_or_baseline(partition_dir, tmp_path, monkeypatch, capsys):
    _write_partition_set(partition_dir)
    legacy_path = tmp_path / "models" / "model_v0.joblib"
    legacy_path.parent.mkdir()
    legacy_path.write_bytes(b"synthetic legacy sentinel; never deserialize")
    legacy_before = _sha(legacy_path.read_bytes())
    baseline_path = tmp_path / "reports" / "baseline_metrics.json"
    baseline_path.parent.mkdir()
    baseline_path.write_text("invalid legacy baseline sentinel", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    artifacts, _report_path, report = _run_all_trainers_and_compare(
        partition_dir, tmp_path / "artifacts"
    )
    capsys.readouterr()

    assert legacy_before == _sha(legacy_path.read_bytes())
    assert baseline_path.read_text(encoding="utf-8") == "invalid legacy baseline sentinel"
    assert report["validation_rows"] == 32
    assert len(artifacts) == 3


def test_training_and_compare_make_no_network_calls(partition_dir, tmp_path, monkeypatch, capsys):
    _write_partition_set(
        partition_dir,
        test_meta={"path": "../holdout/test.csv", "rows": 1, "sha256": "c" * 64},
    )

    def blocked(*_args, **_kwargs):
        raise AssertionError("Network access attempted in the shared-partition workflow.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    _run_all_trainers_and_compare(partition_dir, tmp_path / "artifacts")
    capsys.readouterr()
