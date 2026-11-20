"""Load a saved artifact and predict a URL without contacting the target site."""
from __future__ import annotations

import argparse
import os
import joblib
import pandas as pd
from src.feature_extraction import extract_features_batch

DEFAULT_CANDIDATES = ["models/final_model.joblib", "models/model_v0.joblib", "models/logistic_regression.joblib", "models/tfidf_linear_svm.joblib"]


def find_model_path(path: str | None = None) -> str:
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    for candidate in DEFAULT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("No model found. Train one first; see README.md.")


def predict_url(url: str, model_path: str | None = None) -> dict:
    artifact = joblib.load(find_model_path(model_path))
    model = artifact["model"]
    if artifact["model_type"] == "url_features":
        X = extract_features_batch([url])
        X = X.reindex(columns=artifact.get("feature_columns", X.columns), fill_value=0)
    elif artifact["model_type"] == "url_text":
        X = [url]
    else:
        raise ValueError(f"Unsupported model_type: {artifact.get('model_type')}")
    prediction = int(model.predict(X)[0])
    result = {"prediction": prediction, "label": "Phishing" if prediction == 1 else "Legitimate"}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["probability_legitimate"] = float(probabilities[0])
        result["probability_phishing"] = float(probabilities[1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL string to classify")
    parser.add_argument("--model", default=None, help="Optional model artifact path")
    args = parser.parse_args()
    print(predict_url(args.url, args.model))


if __name__ == "__main__":
    main()
