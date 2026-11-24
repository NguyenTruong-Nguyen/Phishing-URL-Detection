"""Explicit Random Forest training entry point; never overwrites artifacts."""
from __future__ import annotations

import argparse

from src.models.random_forest import build_pipeline
from src.train._common import add_training_args, train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = add_training_args(
        argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter),
        "models/random_forest_candidate.joblib",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    print(train_and_save(build_pipeline(args.random_state), "random_forest", args))


if __name__ == "__main__":
    main()
