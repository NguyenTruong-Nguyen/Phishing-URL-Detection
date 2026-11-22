import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import FEATURE_COLUMNS
from src.models.knn import build_pipeline as build_knn
from src.models.logistic_regression import build_pipeline as build_logistic_regression
from src.models.random_forest import build_pipeline as build_random_forest


URLS = [
    "https://www.example.com/about",
    "https://docs.python.org/3/library/",
    "http://192.168.1.10/login?id=1",
    "http://secure-login.paypa1-verify.xyz/a/b?x=%20",
    "https://example.org/contact",
    "http://verify-account.example.net/session",
]
@pytest.mark.parametrize(
    ("builder", "classifier_type", "uses_scaler"),
    [
        (build_random_forest, RandomForestClassifier, False),
        (build_logistic_regression, LogisticRegression, True),
        (build_knn, KNeighborsClassifier, True),
    ],
)
def test_supported_pipelines_share_the_24_feature_interface_without_fitting(
    tmp_path, builder, classifier_type, uses_scaler
):
    model = builder()
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["classifier"], classifier_type)
    transformer = model.named_steps["url_features"]
    assert transformer.get_feature_names_out().tolist() == FEATURE_COLUMNS
    assert list(transformer.transform(URLS).columns) == FEATURE_COLUMNS
    assert ("scaler" in model.named_steps) is uses_scaler
    if uses_scaler:
        assert isinstance(model.named_steps["scaler"], StandardScaler)

    path = tmp_path / "unfitted_pipeline.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert list(loaded.named_steps) == list(model.named_steps)
    assert loaded.named_steps["classifier"].get_params() == model.named_steps[
        "classifier"
    ].get_params()


def test_knn_requires_positive_neighbor_count():
    with pytest.raises(ValueError, match="positive"):
        build_knn(0)
