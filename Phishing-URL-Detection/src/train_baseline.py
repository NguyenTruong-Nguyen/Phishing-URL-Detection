"""Train URL-feature baselines: Logistic Regression and Random Forest."""
from __future__ import annotations

import argparse
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.preprocessing import add_common_cli_args, load_dataset, split_dataset
from src.feature_extraction import extract_features_batch
from src.evaluation import evaluate_classifier


def main():
    parser = add_common_cli_args(argparse.ArgumentParser())
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    data = load_dataset(args.data, args.url_col, args.label_col)
    X_train, X_test, y_train, y_test = split_dataset(data, args.test_size, args.random_state)
    X_train_f = extract_features_batch(X_train)
    X_test_f = extract_features_batch(X_test)

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=args.random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=args.random_state)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=args.random_state)),
        ]),
    }
    os.makedirs(args.model_dir, exist_ok=True)
    results = []
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train_f, y_train)
        results.append(evaluate_classifier(model, X_test_f, y_test, name))
        artifact = {
            "model": model,
            "model_type": "url_features",
            "feature_columns": list(X_train_f.columns),
            "label_mapping": {"0": "Legitimate", "1": "Phishing"},
        }
        joblib.dump(artifact, os.path.join(args.model_dir, f"{name}.joblib"))
        if name == "logistic_regression":
            joblib.dump(artifact, os.path.join(args.model_dir, "model_v0.joblib"))
    print("\nSaved baseline models to", args.model_dir)


if __name__ == "__main__":
    main()
