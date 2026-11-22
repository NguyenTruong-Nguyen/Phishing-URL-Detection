"""Logistic Regression pipeline for the shared 24 URL features."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import URLFeatureTransformer


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build a scaled Logistic Regression pipeline without fitting it."""
    return Pipeline(
        [
            ("url_features", URLFeatureTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
