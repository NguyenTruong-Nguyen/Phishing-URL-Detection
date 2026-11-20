import pytest
from src.preprocessing import normalize_label


def test_normalize_common_labels():
    assert normalize_label(1) == 1
    assert normalize_label("phishing") == 1
    assert normalize_label(0) == 0
    assert normalize_label("legitimate") == 0


def test_unknown_label_fails_loudly():
    with pytest.raises(ValueError):
        normalize_label("unknown-class")
