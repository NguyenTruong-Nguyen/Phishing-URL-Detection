"""Logistic Regression training entry point."""
from __future__ import annotations

import argparse

from src.models.logistic_regression import build_pipeline
from src.train._common import add_training_args, train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = add_training_args(
        argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter),
        "models/logistic_regression.joblib",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    print(train_and_save(build_pipeline(args.random_state), "logistic_regression", args))


if __name__ == "__main__":
    main()
