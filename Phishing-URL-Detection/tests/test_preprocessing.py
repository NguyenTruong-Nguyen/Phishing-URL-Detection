import pandas as pd
import pytest

from src.preprocessing import (
    LABEL_NAMES,
    inspect_url,
    load_dataset,
    normalize_label,
    registered_domain,
    split_dataset,
)


def test_phiusiil_label_mapping_is_explicit_and_reversed():
    assert normalize_label(1) == 0
    assert normalize_label("1") == 0
    assert normalize_label(0) == 1
    assert normalize_label("0") == 1
    assert LABEL_NAMES == {0: "Legitimate", 1: "Phishing"}
    with pytest.raises(ValueError):
        normalize_label("unknown")


def test_missing_empty_and_whitespace_urls_are_counted_not_stringified(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "URL": [None, float("nan"), "", "   ", "https://valid.example.com"],
            "label": [1, 1, 0, 0, 1],
        }
    ).to_csv(path, index=False, na_rep="NaN")

    result = load_dataset(path)
    audit = result.attrs["audit"]
    assert audit["missing_url"] == 2
    assert audit["empty_url"] == 1
    assert audit["whitespace_url"] == 1
    assert "nan" not in result["url"].tolist()
    assert result["url"].tolist() == ["https://valid.example.com"]


def test_missing_scheme_is_valid_and_original_url_is_preserved():
    assert inspect_url("example.com/path") == ("missing_scheme", False)
    assert inspect_url("https://\u4f8b\u5b50.\u6d4b\u8bd5/\u8def\u5f84")[0] == "valid"
    assert inspect_url("https://example.com:bad/")[0] == "malformed"
    assert inspect_url("http://[broken")[0] == "malformed"


def test_duplicate_urls_same_label_are_deduplicated_and_conflicts_excluded(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "URL": [
                "https://same.example.com/",
                "https://same.example.com/",
                "https://conflict.example.net/",
                "https://conflict.example.net/",
                "https://other.example.org/",
            ],
            "label": [1, 1, 1, 0, 0],
        }
    ).to_csv(path, index=False)

    result = load_dataset(path)
    audit = result.attrs["audit"]
    assert audit["duplicate_url_groups"] == 1
    assert audit["duplicate_excess_rows"] == 1
    assert audit["conflicting_url_groups"] == 1
    assert audit["conflicting_url_rows"] == 2
    assert "https://conflict.example.net/" not in result["url"].tolist()
    assert "https://same.example.com/" in result["url"].tolist()


def test_domain_split_has_no_url_or_registered_domain_overlap():
    rows = []
    for index in range(30):
        label = index % 2
        rows.append({"url": f"https://host{index}.sample{index}.com/login", "label": label})
        rows.append({"url": f"https://www.host{index}.sample{index}.com/account", "label": label})
    frame = pd.DataFrame(rows)

    x_train, x_test, y_train, y_test = split_dataset(frame, test_size=0.25, random_state=7)
    assert set(x_train).isdisjoint(set(x_test))
    train_domains = {registered_domain(url) for url in x_train}
    test_domains = {registered_domain(url) for url in x_test}
    assert train_domains.isdisjoint(test_domains)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_domain_split_fails_explicitly_when_groups_cannot_represent_both_classes():
    frame = pd.DataFrame(
        {
            "url": ["https://a.only.com/", "https://b.only.com/"],
            "label": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="registered-domain split"):
        split_dataset(frame, test_size=0.5)
