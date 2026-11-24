"""Regression coverage for deterministic shared partition export."""
from __future__ import annotations

import hashlib
import csv
import io
import json
import os
import socket
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import load_dataset, registered_domain, split_dataset


# Captured from the original _common.domain_split implementation before refactor.
ORIGINAL_DOMAIN_SPLIT_GOLDEN_SHA256 = (
    "f9e577f1b3c842f2a3ecfb58c7f8afdf5b8863fd9cbcc25490aa986546074a60"
)


def _write_synthetic_source(path: Path) -> None:
    """Write deterministic source-label CSV rows for 600 domains per class."""
    rows = []
    for class_id, (prefix, suffix, source_label) in enumerate(
        (("legit", "com", 1), ("phish", "net", 0))
    ):
        for domain_id in range(600):
            count = 1 + ((domain_id * 7 + class_id * 3) % 5)
            host = f"{prefix}{domain_id:04d}.{suffix}"
            for url_id in range(count):
                scheme = (
                    "https"
                    if (domain_id + url_id + class_id) % 2 == 0
                    else "http"
                )
                path_kind = (domain_id + url_id) % 3
                if path_kind == 0:
                    url_path = "/login"
                elif path_kind == 1:
                    url_path = "/account/reset"
                else:
                    url_path = f"/offers/item_{url_id}"
                rows.append((f"{scheme}://{host}{url_path}", source_label))
    pd.DataFrame(rows, columns=["URL", "label"]).to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n"
    )


def _original_domain_split_reference(data, test_size=0.2, random_state=42, validation_size=0.15):
    """Verbatim copy of domain_split before the _common.py refactor."""
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
        y_fit.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        summary,
    )


def _reference_payload(data):
    x_fit, x_validation, y_fit, y_validation, summary = (
        _original_domain_split_reference(data)
    )
    return {
        "fit_urls": x_fit.tolist(),
        "fit_labels": y_fit.tolist(),
        "validation_urls": x_validation.tolist(),
        "validation_labels": y_validation.tolist(),
        "summary": summary,
    }


def _canonical_hash(payload) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_original_reference_matches_pre_refactor_golden_hash(tmp_path):
    source = tmp_path / "synthetic.csv"
    _write_synthetic_source(source)
    data = load_dataset(str(source))

    assert len(data) == 3600
    assert data.attrs["audit"]["rows_input"] == 3600
    assert _canonical_hash(_reference_payload(data)) == ORIGINAL_DOMAIN_SPLIT_GOLDEN_SHA256


@pytest.fixture(scope="session")
def synthetic_case(tmp_path_factory):
    from src.train.export_partitions import BASELINE_FIELDS

    root = tmp_path_factory.mktemp("shared-partition-source")
    source = root / "synthetic.csv"
    _write_synthetic_source(source)
    data = load_dataset(str(source))
    reference = _reference_payload(data)
    summary = reference["summary"]
    baseline_path = root / "baseline.json"
    baseline_path.write_text(
        json.dumps({"split": {field: summary[field] for field in BASELINE_FIELDS}}),
        encoding="utf-8",
    )
    return {
        "source": source,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline": baseline_path,
        "data": data,
        "reference": reference,
    }


def _run_export(case, shared: Path, holdout: Path, **kwargs):
    from src.train.export_partitions import main

    expected_sha256 = kwargs.pop("expected_sha256", case["source_sha256"])
    baseline_path = kwargs.pop("baseline_path", case["baseline"])
    return main(
        [
            "--data",
            str(case["source"]),
            "--output",
            str(shared),
            "--holdout-output",
            str(holdout),
        ],
        expected_sha256=expected_sha256,
        baseline_path=baseline_path,
        **kwargs,
    )


def _quiet_assert(condition, message):
    if not condition:
        pytest.fail(message)


def _csv_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_refactored_helper_matches_reference_and_golden(synthetic_case):
    from src.train._common import _domain_split_all, domain_split

    data = synthetic_case["data"]
    reference = synthetic_case["reference"]
    result = _domain_split_all(data, 0.2, 42, 0.15)
    x_fit, x_validation, x_test, y_fit, y_validation, y_test, summary = result
    actual = {
        "fit_urls": x_fit.tolist(),
        "fit_labels": y_fit.tolist(),
        "validation_urls": x_validation.tolist(),
        "validation_labels": y_validation.tolist(),
        "summary": summary,
    }
    _quiet_assert(actual == reference, "Refactored split differs from the original ordered reference.")
    _quiet_assert(
        _canonical_hash(actual) == ORIGINAL_DOMAIN_SPLIT_GOLDEN_SHA256,
        "Refactored split does not match the fixed pre-refactor golden hash.",
    )
    _quiet_assert(len(x_test) == summary["test_rows"] == len(y_test), "Test rows and labels are inconsistent.")

    public = domain_split(data, 0.2, 42, 0.15)
    public_payload = {
        "fit_urls": public[0].tolist(),
        "validation_urls": public[1].tolist(),
        "fit_labels": public[2].tolist(),
        "validation_labels": public[3].tolist(),
        "summary": public[4],
    }
    _quiet_assert(public_payload == reference, "Public domain_split behavior or return order changed.")


def test_export_structure_csv_integrity_and_reproducibility(synthetic_case, tmp_path):
    shared_a, holdout_a = tmp_path / "first" / "shared", tmp_path / "first" / "holdout"
    shared_b, holdout_b = tmp_path / "second" / "shared", tmp_path / "second" / "holdout"
    _run_export(synthetic_case, shared_a, holdout_a)
    _run_export(synthetic_case, shared_b, holdout_b)

    expected_shared = {"fit.csv", "validation.csv", "split_manifest.json", "partition_info.txt"}
    _quiet_assert({path.name for path in shared_a.iterdir()} == expected_shared, "Shared output layout is incorrect.")
    _quiet_assert({path.name for path in holdout_a.iterdir()} == {"test.csv"}, "Holdout output layout is incorrect.")
    _quiet_assert(not (shared_a / "test.csv").exists(), "Test CSV was written to the shared directory.")

    data = synthetic_case["data"]
    x_fit, x_validation, x_test, y_fit, y_validation, y_test, _ = __import__(
        "src.train._common", fromlist=["_domain_split_all"]
    )._domain_split_all(data, 0.2, 42, 0.15)
    for path, urls, labels in (
        (shared_a / "fit.csv", x_fit, y_fit),
        (shared_a / "validation.csv", x_validation, y_validation),
        (holdout_a / "test.csv", x_test, y_test),
    ):
        content = path.read_bytes()
        _quiet_assert(not content.startswith(b"\xef\xbb\xbf"), "CSV unexpectedly contains a UTF-8 BOM.")
        _quiet_assert(b"\r" not in content, "CSV line endings are not LF-only.")
        rows = list(csv.reader(io.StringIO(content.decode("utf-8"))))
        _quiet_assert(rows[0] == ["url", "label"], "CSV columns are not exactly url,label.")
        expected_rows = [[url, str(int(label))] for url, label in zip(urls.tolist(), labels.tolist())]
        _quiet_assert(rows[1:] == expected_rows, "CSV row order, URL text, or labels changed.")

    for filename in ("fit.csv", "validation.csv"):
        _quiet_assert(
            _csv_digest(shared_a / filename) == _csv_digest(shared_b / filename),
            "Repeated exports produced different shared CSV hashes.",
        )
    _quiet_assert(
        _csv_digest(holdout_a / "test.csv") == _csv_digest(holdout_b / "test.csv"),
        "Repeated exports produced different holdout CSV hashes.",
    )


def test_manifest_and_partition_info_keep_test_metadata_restricted(synthetic_case, tmp_path):
    shared, holdout = tmp_path / "shared", tmp_path / "holdout"
    _run_export(synthetic_case, shared, holdout)
    manifest = json.loads((shared / "split_manifest.json").read_text(encoding="utf-8"))
    test_meta = manifest["partitions"]["test"]
    _quiet_assert(
        set(test_meta) == {"path", "rows", "sha256", "location"},
        "Shared manifest contains disallowed Test metadata.",
    )
    _quiet_assert("label_distribution" not in test_meta, "Shared manifest exposes Test label distribution.")
    _quiet_assert("registered_domain_count" not in test_meta, "Shared manifest exposes Test domain count.")
    _quiet_assert(manifest["partitions"]["fit"]["sha256"] == _csv_digest(shared / "fit.csv"), "Fit hash mismatch.")
    _quiet_assert(
        manifest["partitions"]["validation"]["sha256"] == _csv_digest(shared / "validation.csv"),
        "Validation hash mismatch.",
    )
    _quiet_assert(test_meta["sha256"] == _csv_digest(holdout / "test.csv"), "Test hash mismatch.")
    _quiet_assert(
        Path(os.path.normpath(str(shared / test_meta["path"]))) == Path(os.path.normpath(str(holdout / "test.csv"))),
        "Manifest Test relative path does not resolve to the holdout CSV.",
    )

    from src.feature_extraction import FEATURE_COLUMNS
    from src.preprocessing import _TLD_EXTRACT

    _quiet_assert(manifest["features"] == {"count": len(FEATURE_COLUMNS), "names": FEATURE_COLUMNS}, "Feature metadata mismatch.")
    _quiet_assert(manifest["environment"]["tldextract_suffix_list_urls"] == [], "Manifest does not record offline suffix extraction.")
    _quiet_assert(tuple(_TLD_EXTRACT.suffix_list_urls) == (), "tldextract is not configured offline.")
    _quiet_assert(set(manifest["code_sha256"]) == {
        "src/preprocessing.py", "src/train/_common.py", "src/train/export_partitions.py"
    }, "Code SHA-256 list is incomplete.")
    for relative, digest in manifest["code_sha256"].items():
        _quiet_assert(
            digest == hashlib.sha256((Path(__file__).resolve().parents[1] / relative).read_bytes()).hexdigest(),
            "Code SHA-256 does not match the source file.",
        )

    comparison = manifest["split"]["baseline_comparison"]
    _quiet_assert(set(comparison) == {
        "fit_rows", "validation_rows", "test_rows", "fit_domains", "validation_domains", "test_domains"
    }, "Historical baseline comparison is incomplete.")
    _quiet_assert(set(comparison["test_domains"]) == {"match"}, "Test domain comparison exposes a count.")
    _quiet_assert(
        all(result["match"] for name, result in comparison.items() if name != "test_domains"),
        "A non-domain historical baseline comparison unexpectedly failed.",
    )
    info = (shared / "partition_info.txt").read_text(encoding="utf-8")
    _quiet_assert("Test registered-domain count" not in info, "Partition info exposes Test domain count.")
    _quiet_assert("Test label distribution" not in info, "Partition info exposes Test label distribution.")
    _quiet_assert("test_domains: match=" in info, "Partition info lacks the restricted Test-domain match flag.")

    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _quiet_assert(manifest["git"]["commit"] == expected_commit, "Manifest Git commit is not current.")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True, text=True,
    ).stdout
    _quiet_assert(manifest["git"]["working_tree_dirty"] == bool(status.strip()), "Manifest dirty flag is not current.")


def test_export_fails_on_missing_or_wrong_hash_before_loading(synthetic_case, tmp_path, monkeypatch):
    from src.train import export_partitions as exporter

    shared, holdout = tmp_path / "missing-shared", tmp_path / "missing-holdout"
    with pytest.raises(exporter.PartitionExportError, match="SHA-256"):
        exporter.main(
            ["--data", str(synthetic_case["source"]), "--output", str(shared), "--holdout-output", str(holdout)],
            expected_sha256="0" * 64,
            baseline_path=synthetic_case["baseline"],
        )
    _quiet_assert(not shared.exists() and not holdout.exists(), "Hash failure left output directories.")

    missing = tmp_path / "not-present.csv"
    with pytest.raises(exporter.PartitionExportError, match="SHA-256"):
        exporter.main(
            ["--data", str(missing), "--output", str(shared), "--holdout-output", str(holdout)],
            expected_sha256=synthetic_case["source_sha256"],
            baseline_path=synthetic_case["baseline"],
        )
    _quiet_assert(not shared.exists() and not holdout.exists(), "Missing dataset left output directories.")


@pytest.mark.parametrize("baseline_kind", ["missing", "invalid_json", "missing_field"])
def test_export_fails_on_invalid_historical_baseline(synthetic_case, tmp_path, baseline_kind):
    from src.train.export_partitions import PartitionExportError, main

    baseline = tmp_path / "bad-baseline.json"
    if baseline_kind == "invalid_json":
        baseline.write_text("{bad", encoding="utf-8")
    elif baseline_kind == "missing_field":
        baseline.write_text(json.dumps({"split": {"fit_rows": 1}}), encoding="utf-8")
    elif baseline_kind == "missing":
        baseline = tmp_path / "absent.json"
    shared, holdout = tmp_path / "shared", tmp_path / "holdout"
    with pytest.raises(PartitionExportError):
        _run_export(synthetic_case, shared, holdout, baseline_path=baseline)
    _quiet_assert(not shared.exists() and not holdout.exists(), "Baseline failure left output directories.")


@pytest.mark.parametrize("audit_kind", ["missing", "missing_rows_input"])
def test_export_fails_on_invalid_audit_metadata(synthetic_case, tmp_path, monkeypatch, audit_kind):
    from src.train import export_partitions as exporter

    data = synthetic_case["data"].copy()
    data.attrs = {}
    if audit_kind == "missing_rows_input":
        data.attrs["audit"] = {"processed_rows": len(data)}
    monkeypatch.setattr(exporter, "load_dataset", lambda _path: data)
    shared, holdout = tmp_path / "shared", tmp_path / "holdout"
    with pytest.raises(exporter.PartitionExportError, match="audit"):
        _run_export(synthetic_case, shared, holdout)
    _quiet_assert(not shared.exists() and not holdout.exists(), "Audit failure left output directories.")


@pytest.mark.parametrize("relation", ["equal", "shared_inside_holdout", "holdout_inside_shared"])
def test_export_rejects_equal_or_nested_output_paths(synthetic_case, tmp_path, relation):
    from src.train.export_partitions import PartitionExportError

    if relation == "equal":
        shared = holdout = tmp_path / "same"
    elif relation == "shared_inside_holdout":
        holdout, shared = tmp_path / "base", tmp_path / "base" / "nested"
    else:
        shared, holdout = tmp_path / "base", tmp_path / "base" / "nested"
    with pytest.raises(PartitionExportError, match="separate"):
        _run_export(synthetic_case, shared, holdout)
    _quiet_assert(not tmp_path.exists() or not any(tmp_path.iterdir()), "Nested-path rejection created output.")


def test_export_refuses_existing_output_and_preserves_existing_files(synthetic_case, tmp_path):
    from src.train.export_partitions import PartitionExportError

    shared, holdout = tmp_path / "shared", tmp_path / "holdout"
    shared.mkdir()
    existing = shared / "fit.csv"
    existing.write_bytes(b"pre-existing")
    with pytest.raises(PartitionExportError, match="overwrite"):
        _run_export(synthetic_case, shared, holdout)
    _quiet_assert(existing.read_bytes() == b"pre-existing", "Existing output was modified.")
    _quiet_assert(not holdout.exists(), "Preflight failure created the holdout directory.")


def test_export_detects_registered_domain_overlap_before_writing(synthetic_case, tmp_path, monkeypatch):
    from src.train import export_partitions as exporter

    controlled_data = pd.DataFrame(
        {
            "url": [
                "https://same.example.com/fit",
                "https://other.example.net/validation",
                "https://same.example.com/test",
            ],
            "label": [0, 1, 0],
        }
    )
    controlled_data.attrs["audit"] = {"rows_input": 3}
    monkeypatch.setattr(exporter, "load_dataset", lambda _path: controlled_data)

    def overlapping(*_args):
        urls_fit = pd.Series(["https://same.example.com/fit"])
        urls_validation = pd.Series(["https://other.example.net/validation"])
        urls_test = pd.Series(["https://same.example.com/test"])
        labels = pd.Series([0])
        return (
            urls_fit, urls_validation, urls_test,
            labels, labels, labels,
            {"fit_rows": 1, "validation_rows": 1, "test_rows": 1,
             "fit_domains": 1, "validation_domains": 1, "test_domains": 1},
        )

    monkeypatch.setattr(exporter, "_domain_split_all", overlapping)
    shared, holdout = tmp_path / "shared", tmp_path / "holdout"
    with pytest.raises(exporter.PartitionExportError, match="overlap"):
        _run_export(synthetic_case, shared, holdout)
    _quiet_assert(not shared.exists() and not holdout.exists(), "Overlap rejection left output directories.")


def test_write_failure_rolls_back_only_current_run_files(synthetic_case, tmp_path, monkeypatch):
    from src.train import export_partitions as exporter

    shared, holdout = tmp_path / "shared", tmp_path / "holdout"
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    original_writer = exporter._write_new_file
    calls = 0

    def fail_second(path, content, created_files):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original_writer(path, content, created_files)

    monkeypatch.setattr(exporter, "_write_new_file", fail_second)
    with pytest.raises(OSError, match="injected write failure"):
        _run_export(synthetic_case, shared, holdout)
    _quiet_assert(unrelated.read_text(encoding="utf-8") == "keep", "Rollback removed an unrelated file.")
    _quiet_assert(not shared.exists() and not holdout.exists(), "Rollback left files or created directories.")


def test_source_dataset_hash_is_unchanged_by_export(synthetic_case, tmp_path):
    before = _csv_digest(synthetic_case["source"])
    _run_export(synthetic_case, tmp_path / "shared", tmp_path / "holdout")
    after = _csv_digest(synthetic_case["source"])
    _quiet_assert(before == after == synthetic_case["source_sha256"], "Exporter changed the source dataset.")


def test_preprocessing_and_split_work_with_network_calls_blocked(synthetic_case, monkeypatch):
    from src.preprocessing import _TLD_EXTRACT

    def blocked(*_args, **_kwargs):
        raise AssertionError("Network access attempted during offline preprocessing.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    _quiet_assert(tuple(_TLD_EXTRACT.suffix_list_urls) == (), "tldextract suffix list must remain offline.")
    data = load_dataset(str(synthetic_case["source"]))
    _original_domain_split_reference(data)
