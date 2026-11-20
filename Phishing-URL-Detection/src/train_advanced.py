"""Train advanced URL models and compare them on the registered-domain validation set."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from scipy import sparse

from src.evaluation import evaluate_classifier
from src.feature_extraction import FEATURE_COLUMNS, URLFeatureTransformer, extract_features_batch
from src.preprocessing import (
    LABEL_NAMES,
    PHIUSIIL_LABEL_MAPPING,
    add_common_cli_args,
    load_dataset,
    registered_domain,
    split_dataset,
)
from src.train_baseline import build_models


TFIDF_CONFIG = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
    "dtype": "float32",
    "lowercase": False,
}

HIST_CONFIGS = {
    "hist_gradient_boosting_100": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
    },
    "hist_gradient_boosting_200": {
        "max_iter": 200,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
    },
    "hist_gradient_boosting_slow": {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
    },
}


def build_hist_pipeline(config: dict, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("url_features", URLFeatureTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    **config,
                    class_weight="balanced",
                    early_stopping=False,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_tfidf_pipeline(classifier: str, random_state: int) -> Pipeline:
    vectorizer = TfidfVectorizer(**{**TFIDF_CONFIG, "dtype": np.float32})
    if classifier == "logistic_regression":
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif classifier == "linear_svm":
        estimator = LinearSVC(
            dual="auto",
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown TF-IDF classifier: {classifier}")
    return Pipeline([("tfidf", vectorizer), ("classifier", estimator)])


def _validation_scores(model, urls: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(urls))[:, list(model.classes_).index(1)]
    return np.asarray(model.decision_function(urls)).ravel()


def _rank(metrics: dict) -> tuple[float, float, float]:
    return metrics["recall"], metrics["f1"], metrics["precision"]


def _plot_validation_reports(results: dict, y_validation, roc_data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(results)
    score_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, metric in enumerate(score_names):
        ax.bar(x + (idx - 1.5) * width, [results[name][metric] for name in names], width, label=metric)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Validation score")
    ax.set_title("Model comparison on registered-domain validation set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "advanced_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, name in zip(axes.flat, names):
        ConfusionMatrixDisplay(
            np.asarray(results[name]["confusion_matrix"]),
            display_labels=["Legitimate", "Phishing"],
        ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(name)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Validation confusion matrices (true rows, predicted columns)")
    fig.tight_layout()
    fig.savefig(output_dir / "advanced_validation_confusion_matrices.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, scores in roc_data.items():
        fpr, tpr, _ = roc_curve(y_validation, scores)
        auc_value = roc_auc_score(y_validation, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC curves")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "advanced_validation_roc_curves.png", dpi=160)
    plt.close(fig)


def _plot_feature_importance(importances: dict[str, float], output_dir: Path):
    ordered = sorted(importances.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([name for name, _ in ordered], [value for _, value in ordered], color="#2878B5")
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Baseline Random Forest: 24 URL feature importances")
    fig.tight_layout()
    fig.savefig(output_dir / "random_forest_feature_importance.png", dpi=160)
    plt.close(fig)


def _save_artifact(model, model_name: str, model_type: str, artifact_path: Path, metadata: dict):
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifact: {artifact_path}")
    staging_path = artifact_path.with_suffix(artifact_path.suffix + ".staging")
    try:
        joblib.dump(
            {
                "artifact_version": 1,
                "model": model,
                "model_type": model_type,
                "model_name": model_name,
                **metadata,
            },
            staging_path,
        )
        loaded = joblib.load(staging_path)
        probe = "https://example.com/login"
        prediction = loaded["model"].predict([probe])
        if len(prediction) != 1:
            raise RuntimeError("Reloaded advanced artifact failed its prediction check.")
        staging_path.replace(artifact_path)
    finally:
        if staging_path.exists():
            staging_path.unlink()


def main():
    parser = add_common_cli_args(argparse.ArgumentParser())
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    if args.split_strategy != "domain":
        parser.error("Advanced comparison requires the verified registered-domain split.")

    data = load_dataset(args.data, args.url_col, args.label_col)
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
        test_size=0.15,
        random_state=args.random_state + 1,
        strategy="domain",
    )
    X_fit, X_validation = X_fit.reset_index(drop=True), X_validation.reset_index(drop=True)
    y_fit, y_validation = y_fit.reset_index(drop=True), y_validation.reset_index(drop=True)
    x_fit, x_validation, x_test = X_fit.tolist(), X_validation.tolist(), X_test.tolist()

    baseline_path = Path(args.reports_dir) / "baseline_metrics.json"
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_split = baseline_report["split"]
    if (len(X_train), len(X_test), len(X_fit), len(X_validation)) != (
        expected_split["train_rows"],
        expected_split["test_rows"],
        expected_split["fit_rows"],
        expected_split["validation_rows"],
    ):
        raise RuntimeError("Recreated split sizes differ from the verified Stage 3A split.")
    model_v0_path = Path(args.model_dir) / "model_v0.joblib"
    if not model_v0_path.is_file():
        raise FileNotFoundError(f"Required baseline artifact is missing: {model_v0_path}")
    model_v0_sha256 = hashlib.sha256(model_v0_path.read_bytes()).hexdigest()

    domain_train = {registered_domain(url) for url in X_train}
    domain_test = {registered_domain(url) for url in X_test}
    domain_fit = {registered_domain(url) for url in X_fit}
    domain_validation = {registered_domain(url) for url in X_validation}
    if domain_train & domain_test or domain_fit & domain_validation:
        raise RuntimeError("Registered-domain leakage detected; refusing advanced training.")
    if set(X_train) & set(X_test) or set(X_fit) & set(X_validation):
        raise RuntimeError("Exact URL leakage detected; refusing advanced training.")
    split_report = {
        "outer_train_rows": len(X_train),
        "outer_test_rows": len(X_test),
        "fit_rows": len(X_fit),
        "validation_rows": len(X_validation),
        "outer_domain_overlap": len(domain_train & domain_test),
        "fit_validation_domain_overlap": len(domain_fit & domain_validation),
        "outer_url_overlap": len(set(X_train) & set(X_test)),
        "fit_validation_url_overlap": len(set(X_fit) & set(X_validation)),
        "train_domains": len(domain_train),
        "test_domains": len(domain_test),
        "fit_domains": len(domain_fit),
        "validation_domains": len(domain_validation),
        "random_state": args.random_state,
        "validation_random_state": args.random_state + 1,
    }
    print("Verified split:", json.dumps(split_report, indent=2))

    results: dict[str, dict] = {}
    roc_data: dict[str, np.ndarray] = {}
    feature_importance = None
    model = None
    gc.collect()

    # Refit both baseline candidates on the same fit domains to create ROC curves
    # and comparable validation predictions; model_v0 itself remains untouched.
    for name, model in build_models(args.random_state).items():
        started = time.perf_counter()
        model.fit(x_fit, y_fit)
        elapsed = time.perf_counter() - started
        metrics = evaluate_classifier(
            model, x_validation, y_validation, name, fit_time_seconds=elapsed, save_plot=False
        )
        results[name] = metrics
        roc_data[name] = _validation_scores(model, x_validation)
        if name == "random_forest":
            importances = model.named_steps["classifier"].feature_importances_
            feature_importance = dict(zip(FEATURE_COLUMNS, map(float, importances)))
        del model
        gc.collect()

    hist_results = {}
    hist_models = {}
    hist_roc_data = {}
    for name, config in HIST_CONFIGS.items():
        model = build_hist_pipeline(config, args.random_state)
        started = time.perf_counter()
        model.fit(x_fit, y_fit)
        elapsed = time.perf_counter() - started
        metrics = evaluate_classifier(
            model, x_validation, y_validation, name, fit_time_seconds=elapsed, save_plot=False
        )
        hist_results[name] = metrics
        hist_models[name] = config
        hist_roc_data[name] = _validation_scores(model, x_validation)
        del model
        gc.collect()
    best_hist_name = max(hist_results, key=lambda name: _rank(hist_results[name]))
    results["hist_gradient_boosting"] = hist_results[best_hist_name]
    results["hist_gradient_boosting"]["selected_config_name"] = best_hist_name
    roc_data["hist_gradient_boosting"] = hist_roc_data[best_hist_name]

    tfidf_results = {}
    tfidf_configs = {}
    for name, classifier in (
        ("tfidf_logistic_regression", "logistic_regression"),
        ("tfidf_linear_svm", "linear_svm"),
    ):
        model = build_tfidf_pipeline(classifier, args.random_state)
        started = time.perf_counter()
        # Pipeline.fit fits the vocabulary on x_fit only; validation is transform-only.
        model.fit(x_fit, y_fit)
        elapsed = time.perf_counter() - started
        vocab_size = len(model.named_steps["tfidf"].vocabulary_)
        if not sparse.issparse(model.named_steps["tfidf"].transform(x_fit[:2])):
            raise RuntimeError("TF-IDF transformer did not return a sparse matrix.")
        if not hasattr(model.named_steps["classifier"], "coef_"):
            raise RuntimeError(f"Unexpected dense/non-linear classifier in {name}.")
        metrics = evaluate_classifier(
            model, x_validation, y_validation, name, fit_time_seconds=elapsed, save_plot=False
        )
        metrics["tfidf_vocabulary_size"] = vocab_size
        metrics["tfidf_config"] = {
            **TFIDF_CONFIG,
            "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
        }
        tfidf_results[name] = metrics
        tfidf_configs[name] = classifier
        roc_data[name] = _validation_scores(model, x_validation)
        del model
        gc.collect()

    results.update(tfidf_results)
    _plot_validation_reports(
        results,
        y_validation,
        roc_data,
        Path(args.reports_dir) / "figures",
    )
    if feature_importance is not None:
        _plot_feature_importance(feature_importance, Path(args.reports_dir) / "figures")

    # Select among the five requested families using validation only. HistGB
    # configuration selection also used validation only.
    selected_name = max(results, key=lambda name: _rank(results[name]))
    if selected_name == "hist_gradient_boosting":
        selected_config = hist_models[best_hist_name]
        final_model = build_hist_pipeline(selected_config, args.random_state)
        model_type = "raw_url_features"
        artifact_name = "hist_gradient_boosting.joblib"
    elif selected_name in tfidf_configs:
        final_model = build_tfidf_pipeline(tfidf_configs[selected_name], args.random_state)
        selected_config = {**TFIDF_CONFIG, "ngram_range": list(TFIDF_CONFIG["ngram_range"])}
        model_type = "url_text"
        artifact_name = f"{selected_name}.joblib"
    elif selected_name == "random_forest":
        final_model = joblib.load(model_v0_path)["model"]
        selected_config = "Existing Stage 3A artifact; unchanged."
        model_type = "raw_url_features"
        artifact_name = None
    else:
        final_model = build_models(args.random_state)[selected_name]
        selected_config = "Stage 3A baseline configuration."
        model_type = "raw_url_features"
        artifact_name = "logistic_regression_advanced.joblib"

    fit_final_seconds = None
    if artifact_name is not None:
        started = time.perf_counter()
        final_model.fit(X_train.tolist(), y_train)
        fit_final_seconds = time.perf_counter() - started
        final_test_metrics = evaluate_classifier(
            final_model,
            x_test,
            y_test,
            f"{selected_name}_test",
            output_dir=str(Path(args.reports_dir) / "figures"),
            fit_time_seconds=fit_final_seconds,
            save_plot=True,
        )
    else:
        # The existing untouched model_v0 was already evaluated on this exact
        # immutable test split in Stage 3A; reuse its recorded final metrics.
        final_test_metrics = baseline_report["final_test_metrics"]
        final_test_metrics["model"] = "random_forest_test (Stage 3A; reused)"
        final_test_metrics["test_evaluation_reused"] = True
        final_model = joblib.load(model_v0_path)["model"]

    test_predictions = np.asarray(final_model.predict(x_test)).astype(int)
    false_negative_mask = (y_test.to_numpy() == 1) & (test_predictions == 0)
    false_negative_urls = np.asarray(x_test, dtype=object)[false_negative_mask]
    false_negative_features = extract_features_batch(false_negative_urls)
    phishing_features = extract_features_batch(np.asarray(x_test, dtype=object)[y_test.to_numpy() == 1])
    fn_analysis = {
        "selected_model": selected_name,
        "false_negatives": int(false_negative_mask.sum()),
        "phishing_test_samples": int((y_test == 1).sum()),
        "false_negative_rate": float(false_negative_mask.sum() / max((y_test == 1).sum(), 1)),
        "false_negative_feature_medians": {
            key: float(value) for key, value in false_negative_features.median().items()
        },
        "all_phishing_test_feature_medians": {
            key: float(value) for key, value in phishing_features.median().items()
        },
    }
    report_dir = Path(args.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "url": false_negative_urls,
            "true_label": 1,
            "predicted_label": 0,
            "model": selected_name,
        }
    ).to_csv(report_dir / "advanced_false_negatives.csv", index=False)

    final_test_cm = np.asarray(final_test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(final_test_cm, display_labels=["Legitimate", "Phishing"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(f"Selected model test confusion matrix: {selected_name}")
    fig.tight_layout()
    fig.savefig(Path(args.reports_dir) / "figures" / "selected_model_test_confusion_matrix.png", dpi=160)
    plt.close(fig)

    selected_test_scores = _validation_scores(final_model, x_test)
    test_fpr, test_tpr, _ = roc_curve(y_test, selected_test_scores)
    test_auc = roc_auc_score(y_test, selected_test_scores)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(test_fpr, test_tpr, label=f"{selected_name} (AUC={test_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Selected model ROC curve on held-out test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(args.reports_dir) / "figures" / "selected_model_test_roc_curve.png", dpi=160)
    plt.close(fig)

    comparison = {
        "dataset_rows": len(data),
        "label_mapping": {"0": "Legitimate", "1": "Phishing"},
        "split": split_report,
        "baseline_model_v0_sha256": model_v0_sha256,
        "tfidf_config": {**TFIDF_CONFIG, "ngram_range": list(TFIDF_CONFIG["ngram_range"])},
        "hist_gradient_boosting_search": {
            name: {"config": hist_models[name], "validation_metrics": hist_results[name]}
            for name in hist_results
        },
        "validation_metrics": results,
        "selected_model": selected_name,
        "selected_config": selected_config,
        "final_test_metrics": final_test_metrics,
        "selected_test_roc_auc": float(test_auc),
        "false_negative_analysis": fn_analysis,
        "feature_importance_random_forest": feature_importance,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    comparison_path = report_dir / "advanced_model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    if artifact_name:
        artifact_path = Path(args.model_dir) / artifact_name
        metadata = {
            "feature_columns": list(FEATURE_COLUMNS) if model_type == "raw_url_features" else [],
            "label_mapping": {
                "source_to_internal": PHIUSIIL_LABEL_MAPPING,
                "internal": {str(label): name for label, name in LABEL_NAMES.items()},
            },
            "training_config": selected_config,
            "split": split_report,
            "metrics": final_test_metrics,
            "library_versions": comparison["library_versions"],
        }
        _save_artifact(final_model, selected_name, model_type, artifact_path, metadata)
        from src.predict import predict_url

        reloaded_result = predict_url("https://example.com/login", str(artifact_path))
        print("Reloaded advanced model probe:", reloaded_result)
        print("Saved advanced artifact:", artifact_path.resolve())

    final_model_v0_sha256 = hashlib.sha256(model_v0_path.read_bytes()).hexdigest()
    if final_model_v0_sha256 != model_v0_sha256:
        raise RuntimeError("models/model_v0.joblib changed during advanced training.")
    print("Verified model_v0 SHA-256 unchanged:", model_v0_sha256)
    print("Selected model:", selected_name)
    print("Final held-out test metrics:", json.dumps(final_test_metrics, indent=2))
    print("False-negative analysis:", json.dumps(fn_analysis, indent=2))
    print("Comparison report:", comparison_path.resolve())


if __name__ == "__main__":
    main()
