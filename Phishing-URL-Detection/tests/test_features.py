from src.feature_extraction import FEATURE_COLUMNS, extract_url_features, extract_features_batch


def test_feature_count_and_names():
    result = extract_url_features("https://www.example.com/a?x=1")
    assert len(result) == 24
    assert list(result.keys()) == FEATURE_COLUMNS


def test_https_and_www():
    result = extract_url_features("https://www.example.com")
    assert result["has_https"] == 1
    assert result["has_www"] == 1


def test_ip_hostname():
    result = extract_url_features("http://192.168.1.10/login")
    assert result["is_ip_hostname"] == 1


def test_batch_shape():
    frame = extract_features_batch(["https://example.com", "http://test.org"])
    assert frame.shape == (2, 24)
