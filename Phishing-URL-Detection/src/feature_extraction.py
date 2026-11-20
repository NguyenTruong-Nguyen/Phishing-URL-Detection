"""Extract 24 lexical/structural features from a URL string only.

No DNS lookup, HTTP request, page fetch, or browser access is performed.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit, parse_qsl

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLUMNS = [
    "url_length", "hostname_length", "count_dots", "count_hyphens",
    "count_underscores", "count_question_marks", "count_equal_signs",
    "count_at_signs", "count_ampersands", "count_percent_signs",
    "count_slashes", "count_digits", "digit_ratio", "special_char_count",
    "has_https", "is_ip_hostname", "has_www", "path_length",
    "path_segment_count", "query_length", "query_parameter_count",
    "subdomain_count", "tld_length", "character_entropy",
]

_SPECIAL_RE = re.compile(r"[^A-Za-z0-9]")


def _parse_url(raw_url: str):
    """Parse safely; add // when a scheme-less URL is supplied."""
    value = str(raw_url or "").strip()
    candidate = value if "://" in value else "//" + value
    try:
        parts = urlsplit(candidate)
        hostname = (parts.hostname or "").lower().strip(".")
        return value, parts, hostname
    except ValueError:
        return value, urlsplit(""), ""


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _is_ip(hostname: str) -> int:
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_url_features(raw_url: str) -> dict[str, float | int]:
    """Return a fixed 24-feature dictionary for one URL."""
    value, parts, hostname = _parse_url(raw_url)
    path = parts.path or ""
    query = parts.query or ""
    labels = [part for part in hostname.split(".") if part]
    digits = sum(char.isdigit() for char in value)
    length = len(value)
    query_params = parse_qsl(query, keep_blank_values=True) if query else []
    tld = labels[-1] if len(labels) >= 2 else ""

    features = {
        "url_length": length,
        "hostname_length": len(hostname),
        "count_dots": value.count("."),
        "count_hyphens": value.count("-"),
        "count_underscores": value.count("_"),
        "count_question_marks": value.count("?"),
        "count_equal_signs": value.count("="),
        "count_at_signs": value.count("@"),
        "count_ampersands": value.count("&"),
        "count_percent_signs": value.count("%"),
        "count_slashes": value.count("/"),
        "count_digits": digits,
        "digit_ratio": digits / length if length else 0.0,
        "special_char_count": len(_SPECIAL_RE.findall(value)),
        "has_https": int(parts.scheme.lower() == "https"),
        "is_ip_hostname": _is_ip(hostname),
        "has_www": int(hostname.startswith("www.")),
        "path_length": len(path),
        "path_segment_count": len([p for p in path.split("/") if p]),
        "query_length": len(query),
        "query_parameter_count": len(query_params),
        "subdomain_count": max(len(labels) - 2, 0),
        "tld_length": len(tld),
        "character_entropy": _entropy(value),
    }
    return features


def extract_features_batch(urls) -> pd.DataFrame:
    """Convert an iterable of URLs to a DataFrame with stable column order."""
    rows = [extract_url_features(url) for url in urls]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS).fillna(0)


class URLFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer from raw URL strings to the 24 URL features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        return extract_features_batch(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_COLUMNS, dtype=object)
