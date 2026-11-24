"""KNN training entry point with a bounded validation prediction-time probe."""
from __future__ import annotations

import argparse

from src.models.knn import build_pipeline
from src.train._common import add_training_args, train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = add_training_args(
        argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter),
        "models/knn.joblib",
    )
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--timing-sample-size", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timing_sample_size < 1:
        parser.error("--timing-sample-size must be positive")
    result = train_and_save(
        build_pipeline(args.neighbors),
        "knn",
        args,
        timing_sample_size=args.timing_sample_size,
    )
    print(result)


if __name__ == "__main__":
    main()
