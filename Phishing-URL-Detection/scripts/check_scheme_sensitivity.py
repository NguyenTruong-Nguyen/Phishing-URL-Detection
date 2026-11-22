import json
from pathlib import Path

import numpy as np

from src.predict import predict_urls
from src.preprocessing import inspect_url, load_dataset, split_dataset


DATA = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = Path("reports/scheme_sensitivity_report.json")

SAMPLE_PER_CLASS = 1000


def strip_scheme(url: str) -> str:
    return url.split("://", 1)[1] if "://" in url else url


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {OUT}"
        )

    data = load_dataset(DATA)

    _, x_test, _, y_test = split_dataset(
        data,
        0.2,
        42,
        strategy="domain",
    )

    rng = np.random.default_rng(0)

    rows = {}

    for label, name in (
        (0, "Legitimate"),
        (1, "Phishing"),
    ):
        raw_pool = x_test[y_test.eq(label)].tolist()
        pool = []
        excluded = 0

        for url in raw_pool:
            stripped_url = strip_scheme(url)
            original_status, _ = inspect_url(url)
            stripped_status, _ = inspect_url(stripped_url)
            if (
                original_status in {"valid", "missing_scheme"}
                and stripped_status in {"valid", "missing_scheme"}
            ):
                pool.append((url, stripped_url))
            else:
                excluded += 1

        if not pool:
            raise ValueError(
                f"No valid test samples found for class {name}"
            )

        pick = rng.choice(
            len(pool),
            size=min(SAMPLE_PER_CLASS, len(pool)),
            replace=False,
        )

        urls = [pool[i][0] for i in pick]
        stripped = [pool[i][1] for i in pick]

        with_scheme = predict_urls(urls)
        without_scheme = predict_urls(stripped)

        pred_a = np.array([
            result["prediction"]
            for result in with_scheme
        ])

        pred_b = np.array([
            result["prediction"]
            for result in without_scheme
        ])

        rows[name] = {
            "n": len(urls),
            "excluded_after_scheme_strip": excluded,
            "phishing_rate_with_scheme": float(
                pred_a.mean()
            ),
            "phishing_rate_without_scheme": float(
                pred_b.mean()
            ),
            "flip_rate": float(
                (pred_a != pred_b).mean()
            ),
            "accuracy_with_scheme": float(
                (pred_a == label).mean()
            ),
            "accuracy_without_scheme": float(
                (pred_b == label).mean()
            ),
        }

    OUT.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("Saved:", OUT.resolve())


if __name__ == "__main__":
    main()
