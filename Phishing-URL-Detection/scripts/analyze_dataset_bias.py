import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.feature_extraction import extract_features_batch
from src.preprocessing import LABEL_NAMES, load_dataset


DATA = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = Path("reports/dataset_bias_report.json")


def is_bare_homepage(url):
    parsed = urlsplit(url if "://" in url else "//" + url)

    return int(
        bool(parsed.hostname)
        and parsed.path in ("", "/")
        and not parsed.query
    )


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {OUT}"
        )

    data = load_dataset(DATA)

    feats = extract_features_batch(data["url"].tolist())

    feats["label"] = data["label"].to_numpy()

    feats["has_path"] = (
        feats["path_length"] > 1
    ).astype(int)

    feats["has_query"] = (
        feats["query_length"] > 0
    ).astype(int)

    feats["is_bare_homepage"] = (
        data["url"].map(is_bare_homepage).to_numpy()
    )

    report = {
        "rows": int(len(data)),
        "label_distribution": data.attrs["audit"]["label_distribution"],
        "by_label": {},
    }

    for label, name in LABEL_NAMES.items():
        group = feats[feats["label"] == label]

        report["by_label"][name] = {
            "rows": int(len(group)),
            "url_length_mean": float(group["url_length"].mean()),
            "url_length_median": float(group["url_length"].median()),
            "share_https": float(group["has_https"].mean()),
            "share_www": float(group["has_www"].mean()),
            "share_has_path": float(group["has_path"].mean()),
            "share_has_query": float(group["has_query"].mean()),
            "share_bare_homepage": float(
                group["is_bare_homepage"].mean()
            ),
            "share_ip_hostname": float(
                group["is_ip_hostname"].mean()
            ),
            "digit_ratio_mean": float(
                group["digit_ratio"].mean()
            ),
        }

    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(pd.DataFrame(report["by_label"]).T.to_string())
    print("Saved:", OUT.resolve())


if __name__ == "__main__":
    main()
