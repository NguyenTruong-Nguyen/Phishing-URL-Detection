"""KNN training entry point with a bounded validation prediction-time probe."""
from __future__ import annotations

import argparse
import time

from src.models.knn import build_pipeline
from src.preprocessing import load_dataset
from src.train._common import add_training_args, domain_split, train_and_save


def main():
    parser = add_training_args(argparse.ArgumentParser(), "models/knn.joblib")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--timing-sample-size", type=int, default=1000)
    args = parser.parse_args()
    if args.timing_sample_size < 1:
        parser.error("--timing-sample-size must be positive")

    result = train_and_save(build_pipeline(args.neighbors), "knn", args)
    # Measure a bounded sample from Validation to avoid a multi-billion-distance
    # full-set benchmark; validation data is never used for fitting.
    data = load_dataset(args.data, args.url_col, args.label_col)
    _, x_validation, _, _, _ = domain_split(
        data, args.test_size, args.random_state, args.validation_size
    )
    import joblib

    model = joblib.load(args.output)["model"]
    sample = x_validation.iloc[: args.timing_sample_size].tolist()
    started = time.perf_counter()
    model.predict(sample)
    elapsed = time.perf_counter() - started
    result["prediction_timing"] = {
        "sample_size": len(sample),
        "elapsed_seconds": elapsed,
        "ms_per_url": elapsed * 1000 / max(len(sample), 1),
        "note": "bounded validation sample; test partition was not scored",
    }
    print(result)


if __name__ == "__main__":
    main()
