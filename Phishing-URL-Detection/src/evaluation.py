"""Evaluation helpers and report artifact generation."""
from __future__ import annotations

import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)


def evaluate_classifier(model, X_test, y_test, model_name: str, output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    predictions = model.predict(X_test)
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    else:
        metrics["roc_auc"] = None

    start = time.perf_counter()
    model.predict(X_test)
    elapsed = time.perf_counter() - start
    metrics["prediction_ms_per_url"] = float((elapsed / max(len(X_test), 1)) * 1000)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nClassification report:\n", classification_report(y_test, predictions, target_names=["Legitimate", "Phishing"], zero_division=0))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"), dpi=160)
    plt.close()
    return metrics
