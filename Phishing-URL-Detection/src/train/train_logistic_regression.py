"""Logistic Regression training entry point."""
from __future__ import annotations

import argparse

from src.models.logistic_regression import build_pipeline
from src.train._common import add_training_args, train_and_save


def main():
    parser = add_training_args(
        argparse.ArgumentParser(), "models/logistic_regression.joblib"
    )
    args = parser.parse_args()
    print(train_and_save(build_pipeline(args.random_state), "logistic_regression", args))


if __name__ == "__main__":
    main()
