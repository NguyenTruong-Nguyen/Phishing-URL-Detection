"""Train and compare Logistic Regression and Random Forest on raw URL features."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_classifier
from src.feature_extraction import FEATURE_COLUMNS, URLFeatureTransformer
from src.preprocessing import (
    LABEL_NAMES,
    PHIUSIIL_LABEL_MAPPING,
    add_common_cli_args,
    load_dataset,
    registered_domain,
    split_dataset,
)


def build_models(random_state: int) -> dict[str, Pipeline]:
    """Return fixed, reproducible candidate pipelines from raw URL strings."""
    return {
        "logistic_regression": Pipeline(
            [
                ("url_features", URLFeatureTransformer()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("url_features", URLFeatureTransformer()),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _domain_count(urls: pd.Series) -> int:
    return int(urls.map(registered_domain).nunique())


def main():
    parser = add_common_cli_args(argparse.ArgumentParser())
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    if args.split_strategy != "domain":
        parser.error("Stage 3A requires the verified registered-domain split.")

    data = load_dataset(args.data, args.url_col, args.label_col)
    print("Dataset audit:", json.dumps(data.attrs["audit"], ensure_ascii=False))

    # This recreates the already verified deterministic outer split. The test
    # partition is kept untouched until the validation-selected model is final.
    X_train, X_test, y_train, y_test = split_dataset(
        data, args.test_size, args.random_state, strategy="domain"
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    train_frame = pd.DataFrame({"url": X_train.to_numpy(), "label": y_train.to_numpy()})
    X_fit, X_validation, y_fit, y_validation = split_dataset(
        train_frame,
        test_size=args.validation_size,
        random_state=args.random_state + 1,
        strategy="domain",
    )
    y_fit = y_fit.reset_index(drop=True)
    y_validation = y_validation.reset_index(drop=True)

    split_summary = {
        "outer_strategy": "registered_domain",
        "random_state": args.random_state,
        "test_size_requested": args.test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_domains": _domain_count(X_train),
        "test_domains": _domain_count(X_test),
        "outer_domain_overlap": len(
            set(X_train.map(registered_domain)) & set(X_test.map(registered_domain))
        ),
        "fit_rows": int(len(X_fit)),
        "validation_rows": int(len(X_validation)),
        "fit_domains": _domain_count(X_fit),
        "validation_domains": _domain_count(X_validation),
        "fit_validation_domain_overlap": len(
            set(X_fit.map(registered_domain)) & set(X_validation.map(registered_domain))
        ),
    }
    if split_summary["outer_domain_overlap"] or split_summary["fit_validation_domain_overlap"]:
        raise RuntimeError("Registered-domain leakage detected; refusing to train.")
    print("Split summary:", json.dumps(split_summary, indent=2))
    print("Train labels:", y_train.value_counts().sort_index().to_dict())
    print("Test labels:", y_test.value_counts().sort_index().to_dict())
    print("Fit labels:", y_fit.value_counts().sort_index().to_dict())
    print("Validation labels:", y_validation.value_counts().sort_index().to_dict())

    candidates = build_models(args.random_state)
    candidate_results = {}
    for name, model in candidates.items():
        print(f"\n=== Fitting {name} on fit domains ===")
        started = time.perf_counter()
        model.fit(X_fit.tolist(), y_fit)
        fit_seconds = time.perf_counter() - started
        fit_metrics = evaluate_classifier(
            model,
            X_fit.tolist(),
            y_fit,
            f"{name}_fit",
            fit_time_seconds=fit_seconds,
            save_plot=False,
        )
        validation_metrics = evaluate_classifier(
            model,
            X_validation.tolist(),
            y_validation,
            f"{name}_validation",
            fit_time_seconds=fit_seconds,
            save_plot=False,
        )
        candidate_results[name] = {
            "fit": fit_metrics,
            "validation": validation_metrics,
            "fit_seconds": float(fit_seconds),
        }

    # Model choice uses validation only, prioritizing the phishing-class recall.
    selected_name = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["validation"]["recall"],
            candidate_results[name]["validation"]["f1"],
            candidate_results[name]["validation"]["precision"],
        ),
    )
    print("Selected by validation (recall, then F1, then precision):", selected_name)

    # Refit the selected architecture on all outer-training domains. Only now is
    # the independent outer test used for final reporting.
    final_model = build_models(args.random_state)[selected_name]
    started = time.perf_counter()
    final_model.fit(X_train.tolist(), y_train.reset_index(drop=True))
    final_fit_seconds = time.perf_counter() - started
    final_train_metrics = evaluate_classifier(
        final_model,
        X_train.tolist(),
        y_train.reset_index(drop=True),
        f"{selected_name}_final_train",
        fit_time_seconds=final_fit_seconds,
        save_plot=False,
    )
    final_test_metrics = evaluate_classifier(
        final_model,
        X_test.tolist(),
        y_test.reset_index(drop=True),
        f"{selected_name}_test",
        output_dir=str(Path(args.reports_dir) / "figures"),
        fit_time_seconds=final_fit_seconds,
        save_plot=True,
    )

    report = {
        "selected_model": selected_name,
        "selection_metric": "validation phishing recall, then F1, then precision",
        "split": split_summary,
        "candidate_metrics": candidate_results,
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
    }
    report_dir = Path(args.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "baseline_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    artifact = {
        "artifact_version": 1,
        "model": final_model,
        "model_type": "raw_url_features",
        "model_name": selected_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_mapping": {
            "source_to_internal": dict(PHIUSIIL_LABEL_MAPPING),
            "internal": {str(label): name for label, name in LABEL_NAMES.items()},
        },
        "source_columns_used": [args.url_col, args.label_col],
        "split": split_summary,
        "training_config": {
            "random_state": args.random_state,
            "validation_size": args.validation_size,
            "class_weight_logistic_regression": "balanced",
            "class_weight_random_forest": "balanced_subsample",
            "validation_selection": "phishing recall, then F1, then precision",
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "metrics": report,
    }
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "model_v0.joblib"
    joblib.dump(artifact, artifact_path)

    # Reload from disk and verify inference on actual held-out URLs plus the
    # required scheme-less, invalid-input, and batch cases. No HTTP is used.
    from src.predict import predict_url, predict_urls

    reloaded = joblib.load(artifact_path)
    if reloaded["feature_columns"] != list(FEATURE_COLUMNS):
        raise RuntimeError("Saved artifact feature schema does not match the 24 URL features.")
    examples = {}
    for label, name in LABEL_NAMES.items():
        matching = X_test[y_test.eq(label)]
        if matching.empty:
            raise RuntimeError(f"No held-out example for {name}.")
        examples[name] = matching.iloc[0]
    probe_urls = list(examples.values())
    print("Reloaded artifact prediction probes:")
    for true_label, url in examples.items():
        print({"expected_from_dataset": true_label, **predict_url(url, str(artifact_path))})

    scheme_less = probe_urls[0].removeprefix("https://").removeprefix("http://")
    print("Scheme-less probe:", predict_url(scheme_less, str(artifact_path)))
    try:
        predict_url("not a valid URL", str(artifact_path))
    except ValueError as exc:
        print("Invalid URL rejected:", str(exc))
    else:
        raise RuntimeError("Invalid URL unexpectedly passed prediction validation.")
    batch_results = predict_urls(probe_urls + [scheme_less], str(artifact_path))
    if len(batch_results) != len(probe_urls) + 1:
        raise RuntimeError("Batch prediction returned an unexpected number of results.")
    print("Batch prediction count:", len(batch_results))
    print("Saved artifact:", artifact_path.resolve())
    print("Saved metrics:", report_path.resolve())


if __name__ == "__main__":
    main()
