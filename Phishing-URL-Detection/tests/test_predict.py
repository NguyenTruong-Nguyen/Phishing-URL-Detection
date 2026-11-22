import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from src.feature_extraction import FEATURE_COLUMNS, URLFeatureTransformer
from src.predict import find_model_path, predict_url, predict_urls


class FixedClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic inference stub; avoids fitting an estimator in tests."""

    def __init__(self):
        self.classes_ = np.asarray([0, 1])

    def fit(self, X, y=None):
        raise AssertionError("The prediction test must not fit an estimator.")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile(np.asarray([[0.75, 0.25]]), (len(X), 1))


@pytest.fixture()
def model_path(tmp_path):
    pipe = Pipeline([
        ("features", URLFeatureTransformer()),
        ("clf", FixedClassifier()),
    ])

    path = tmp_path / "tiny.joblib"

    joblib.dump(
        {
            "model": pipe,
            "model_type": "raw_url_features",
            "feature_columns": FEATURE_COLUMNS,
        },
        path,
    )

    return str(path)


def test_single_prediction_shape(model_path):
    r = predict_url("https://www.example.com", model_path)

    assert r["label"] in {"Legitimate", "Phishing"}
    assert r["prediction"] in {0, 1}

    assert (
        r["probability_legitimate"] + r["probability_phishing"]
        == pytest.approx(1.0)
    )


def test_scheme_less_url_is_accepted(model_path):
    assert predict_url("www.example.com/login", model_path)["label"]


def test_batch_returns_one_result_per_url(model_path):
    assert len(
        predict_urls(["https://a.com", "http://b.org/x"], model_path)
    ) == 2


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not a valid URL", "http://[broken"],
)
def test_invalid_urls_raise_value_error(model_path, bad):
    with pytest.raises(ValueError):
        predict_url(bad, model_path)


def test_non_string_input_raises(model_path):
    with pytest.raises(ValueError):
        predict_urls([123], model_path)


def test_default_artifact_is_existing_random_forest():
    assert find_model_path().endswith("models\\model_v0.joblib")
