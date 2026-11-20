"""Raw-URL prediction interface; never contacts the target website."""
from __future__ import annotations

import argparse
import os
import time

import joblib

from src.feature_extraction import extract_features_batch
from src.preprocessing import inspect_url

DEFAULT_CANDIDATES = [
    "models/final_model.joblib",
    "models/model_v0.joblib",
    "models/logistic_regression.joblib",
    "models/tfidf_linear_svm.joblib",
]


def find_model_path(path: str | None = None) -> str:
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    for candidate in DEFAULT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("No model found. Train one first; see README.md.")


def load_model(path: str | None = None) -> dict:
    """Load a model artifact once for single or batch inference."""
    artifact = joblib.load(find_model_path(path))
    if not isinstance(artifact, dict) or "model" not in artifact or "model_type" not in artifact:
        raise ValueError("Model artifact has an unsupported structure.")
    return artifact


def _prepare_input(url: str, index: int) -> str:
    if not isinstance(url, str):
        raise ValueError(f"URL at position {index} must be a string.")
    status, _ = inspect_url(url)
    if status not in {"valid", "missing_scheme"}:
        raise ValueError(f"Invalid URL at position {index}: {status}.")
    return url


def predict_urls(urls, model_path: str | None = None) -> list[dict]:
    """Predict multiple raw URLs with one artifact load and one batch call."""
    if isinstance(urls, str):
        urls = [urls]
    urls = [_prepare_input(url, index) for index, url in enumerate(urls)]
    if not urls:
        return []

    artifact = load_model(model_path)
    model = artifact["model"]
    model_type = artifact["model_type"]
    if model_type in {"raw_url_features", "url_text"}:
        X = urls
    elif model_type == "url_features":
        X = extract_features_batch(urls)
        X = X.reindex(columns=artifact.get("feature_columns", X.columns), fill_value=0)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    start = time.perf_counter()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    elapsed_ms_per_url = (time.perf_counter() - start) * 1000 / len(urls)
    classes = list(getattr(model, "classes_", []))
    results = []
    for index, (url, raw_prediction) in enumerate(zip(urls, predictions)):
        prediction = int(raw_prediction)
        result = {
            "url": url,
            "prediction": prediction,
            "label": "Phishing" if prediction == 1 else "Legitimate",
            "processing_time_ms": float(elapsed_ms_per_url),
        }
        if probabilities is not None:
            if classes:
                probability_by_class = dict(zip(classes, probabilities[index]))
                result["probability_legitimate"] = float(probability_by_class.get(0, 0.0))
                result["probability_phishing"] = float(probability_by_class.get(1, 0.0))
            else:
                result["probability_legitimate"] = float(probabilities[index][0])
                result["probability_phishing"] = float(probabilities[index][1])
        results.append(result)
    return results


def predict_url(url: str, model_path: str | None = None) -> dict:
    """Predict one URL and return a stable result dictionary."""
    return predict_urls([url], model_path=model_path)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL string to classify")
    parser.add_argument("--model", default=None, help="Optional model artifact path")
    args = parser.parse_args()
    print(predict_url(args.url, args.model))


if __name__ == "__main__":
    main()
