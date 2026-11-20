"""Train character TF-IDF + Linear SVM URL text classifier."""
from __future__ import annotations

import argparse
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import add_common_cli_args, load_dataset, split_dataset, registered_domain
from src.evaluation import evaluate_classifier


def main():
    parser = add_common_cli_args(argparse.ArgumentParser())
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    data = load_dataset(args.data, args.url_col, args.label_col)
    print("Dataset audit:", data.attrs["audit"])
    X_train, X_test, y_train, y_test = split_dataset(
        data, args.test_size, args.random_state, strategy=args.split_strategy
    )
    print("Split strategy:", args.split_strategy)
    print("Train labels:", y_train.value_counts().sort_index().to_dict())
    print("Test labels:", y_test.value_counts().sort_index().to_dict())
    print("Train registered domains:", X_train.map(registered_domain).nunique())
    print("Test registered domains:", X_test.map(registered_domain).nunique())
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=300000, sublinear_tf=True)),
        ("classifier", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=args.random_state), cv=3)),
    ])
    model.fit(X_train, y_train)
    evaluate_classifier(model, X_test, y_test, "tfidf_linear_svm")
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump({
        "model": model,
        "model_type": "url_text",
        "label_mapping": {"0": "Legitimate", "1": "Phishing"},
    }, os.path.join(args.model_dir, "tfidf_linear_svm.joblib"))
    print("Saved TF-IDF model.")


if __name__ == "__main__":
    main()
