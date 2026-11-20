"""Dataset validation, PhiUSIIL label mapping, and leakage-aware splitting."""
from __future__ import annotations

import argparse
import ipaddress
from functools import lru_cache
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tldextract


# Verified against the official PhiUSIIL dataset description and this CSV:
# source 1 = legitimate, source 0 = phishing; internal 0/1 uses reverse polarity.
PHIUSIIL_LABEL_MAPPING = {"0": 1, "1": 0}
LABEL_NAMES = {0: "Legitimate", 1: "Phishing"}
URL_COLUMN = "URL"
LABEL_COLUMN = "label"
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())  # Never fetch the PSL.


def normalize_label(value) -> int:
    """Map the verified PhiUSIIL source label to 0=Legitimate, 1=Phishing."""
    if pd.isna(value):
        raise ValueError("Missing label cannot be normalized.")
    key = str(value).strip().lower()
    if key in PHIUSIIL_LABEL_MAPPING:
        return PHIUSIIL_LABEL_MAPPING[key]
    if key in {"legitimate", "benign", "safe"}:
        return 0
    if key in {"phishing", "malicious", "fraud", "unsafe"}:
        return 1
    raise ValueError(f"Unknown label {value!r}; refusing to infer its meaning.")


def inspect_url(raw_url) -> tuple[str, bool]:
    """Return (status, has_scheme) without modifying the original URL value."""
    if pd.isna(raw_url):
        return "missing", False
    if not isinstance(raw_url, str):
        raw_url = str(raw_url)
    if raw_url.strip().casefold() in {"nan", "null", "none", "<na>"}:
        return "missing", False
    if raw_url == "":
        return "empty", False
    candidate = raw_url.strip()
    if not candidate:
        return "whitespace", False
    if candidate != raw_url:
        return "malformed", False
    if any(char.isspace() for char in candidate):
        return "malformed", False

    has_scheme = "://" in candidate
    try:
        parts = urlsplit(candidate if has_scheme else "//" + candidate)
        hostname = parts.hostname
        # Accessing .port validates malformed/out-of-range port values.
        _ = parts.port
        if not hostname or "\x00" in hostname:
            return "malformed", has_scheme
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            # IDNs are valid lexical hosts; reject only clearly invalid host syntax.
            if "." not in hostname and hostname.lower() != "localhost":
                return "malformed", has_scheme
        return ("valid" if has_scheme else "missing_scheme"), has_scheme
    except (ValueError, UnicodeError):
        return "malformed", has_scheme


@lru_cache(maxsize=500_000)
def registered_domain(raw_url: str) -> str:
    """Extract an eTLD+1 from the URL text only, with no DNS or network access."""
    status, _ = inspect_url(raw_url)
    if status not in {"valid", "missing_scheme"}:
        return "invalid:" + str(raw_url)
    candidate = raw_url.strip()
    parts = urlsplit(candidate if "://" in candidate else "//" + candidate)
    host = (parts.hostname or "").lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
        return "ip:" + host
    except ValueError:
        pass
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return "host:" + host
    extracted = _TLD_EXTRACT(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    # Unknown/private suffixes remain grouped by hostname, never by whole URL.
    return "host:" + host


def load_dataset(
    path: str,
    url_col: str = URL_COLUMN,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Read PhiUSIIL, audit URL/label issues, and return clean raw URLs.

    The source file is read-only. URL strings are not stripped or rewritten in
    the returned frame. Missing/malformed URLs and contradictory duplicate URL
    groups are excluded; same-label exact duplicates are reduced to one row.
    Audit counts are available as ``result.attrs["audit"]``.
    """
    source = pd.read_csv(path, dtype={url_col: "string"}, keep_default_na=False)
    missing_columns = [name for name in (url_col, label_col) if name not in source.columns]
    if missing_columns:
        raise ValueError(
            f"Missing column(s): {missing_columns}. Available columns: {list(source.columns)}"
        )

    audit = {
        "rows_input": int(len(source)),
        "url_column": url_col,
        "label_column": label_col,
        "missing_url": 0,
        "empty_url": 0,
        "whitespace_url": 0,
        "malformed_url": 0,
        "missing_scheme_url": 0,
        "unicode_url": 0,
        "missing_label": 0,
        "duplicate_url_groups": 0,
        "duplicate_excess_rows": 0,
        "conflicting_url_groups": 0,
        "conflicting_url_rows": 0,
    }
    rows: list[dict] = []
    for raw_url, raw_label in zip(source[url_col], source[label_col]):
        status, _has_scheme = inspect_url(raw_url)
        if status in {"missing", "empty", "whitespace", "malformed"}:
            audit[f"{status}_url"] += 1
            continue
        if status == "missing_scheme":
            audit["missing_scheme_url"] += 1
        if any(ord(char) > 127 for char in str(raw_url)):
            audit["unicode_url"] += 1
        if pd.isna(raw_label) or not str(raw_label).strip():
            audit["missing_label"] += 1
            continue
        rows.append({"url": str(raw_url), "label": normalize_label(raw_label)})

    frame = pd.DataFrame(rows, columns=["url", "label"])
    if frame.empty:
        raise ValueError("No usable URL/label rows remain after validation.")

    group = frame.groupby("url", sort=False)["label"]
    label_counts = group.nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)
    audit["conflicting_url_groups"] = len(conflicting_urls)
    audit["conflicting_url_rows"] = int(frame["url"].isin(conflicting_urls).sum())
    frame = frame[~frame["url"].isin(conflicting_urls)].copy()

    sizes = frame.groupby("url", sort=False).size()
    audit["duplicate_url_groups"] = int((sizes > 1).sum())
    audit["duplicate_excess_rows"] = int((sizes[sizes > 1] - 1).sum())
    frame = frame.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
    audit["rows_after_processing"] = int(len(frame))
    audit["label_distribution"] = {
        LABEL_NAMES[int(label)]: int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }
    frame.attrs["audit"] = audit
    return frame


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: str = "domain",
    attempts: int = 30,
):
    """Split rows, defaulting to registered-domain isolation.

    For domain splitting, deterministic candidate group splits are searched and
    the feasible split closest to requested size/class proportions is returned.
    If no candidate contains both classes in both sets, fail explicitly rather
    than silently falling back to a URL-level random split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    urls = df["url"].astype(str).reset_index(drop=True)
    labels = df["label"].astype(int).reset_index(drop=True)
    if labels.nunique() < 2:
        raise ValueError("Train/test splitting requires both label classes.")
    if strategy == "random":
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            range(len(df)), test_size=test_size, random_state=random_state, stratify=labels
        )
    elif strategy == "domain":
        groups = urls.map(registered_domain)
        group_count = groups.nunique()
        if group_count < 2:
            raise ValueError(
                "Could not create a registered-domain split: at least two "
                "registered domains are required."
            )
        class_group_counts = pd.DataFrame({"label": labels, "domain": groups})
        class_group_counts = class_group_counts.groupby("label")["domain"].nunique()
        if (class_group_counts < 2).any():
            raise ValueError(
                "Could not create a registered-domain split: each label must "
                "occur in at least two registered domains."
            )
        splitter = GroupShuffleSplit(
            n_splits=attempts, test_size=test_size, random_state=random_state
        )
        overall_rate = float(labels.mean())
        best, best_score = None, float("inf")
        try:
            for train_idx, test_idx in splitter.split(urls, labels, groups):
                if labels.iloc[train_idx].nunique() < 2 or labels.iloc[test_idx].nunique() < 2:
                    continue
                test_rate = float(labels.iloc[test_idx].mean())
                size_error = abs(len(test_idx) / len(df) - test_size)
                score = size_error + abs(test_rate - overall_rate)
                if score < best_score:
                    best, best_score = (train_idx, test_idx), score
        except ValueError as exc:
            raise ValueError(
                "Could not create a registered-domain split with the requested "
                f"test size: {exc}"
            ) from exc
        if best is None:
            raise ValueError(
                "Could not create a registered-domain split containing both labels "
                "in train and test. Inspect domain/class distribution; choose an "
                "explicit alternative rather than silently using a random split."
            )
        train_idx, test_idx = best
    else:
        raise ValueError("strategy must be 'domain' or 'random'.")

    return (
        urls.iloc[train_idx],
        urls.iloc[test_idx],
        labels.iloc[train_idx],
        labels.iloc[test_idx],
    )


def add_common_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data", default="data/raw/PhiUSIIL_Phishing_URL_Dataset.csv", help="Input CSV path")
    parser.add_argument("--url-col", default=URL_COLUMN, help="Verified PhiUSIIL URL column")
    parser.add_argument("--label-col", default=LABEL_COLUMN, help="Verified PhiUSIIL label column")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-strategy", choices=("domain", "random"), default="domain")
    return parser
