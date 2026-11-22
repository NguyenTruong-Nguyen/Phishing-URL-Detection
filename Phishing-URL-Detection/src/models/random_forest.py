"""Random Forest pipeline for the shared 24 URL features."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.feature_extraction import URLFeatureTransformer


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build the baseline Random Forest pipeline without fitting it."""
    return Pipeline(
        [
            ("url_features", URLFeatureTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    max_depth=None,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    bootstrap=True,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
