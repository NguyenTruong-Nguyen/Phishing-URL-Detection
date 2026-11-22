"""K-Nearest Neighbors pipeline for the shared 24 URL features."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import URLFeatureTransformer


def build_pipeline(n_neighbors: int = 5) -> Pipeline:
    """Build a scaled KNN pipeline without fitting it."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive.")
    return Pipeline(
        [
            ("url_features", URLFeatureTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)),
        ]
    )
