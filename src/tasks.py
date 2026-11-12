"""Clustering and dimensionality reduction on the UCI Wine Quality dataset."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    model_validator,
    validate_call,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CSV_SEPARATOR: Final[str] = ";"
TYPE_COLUMN: Final[str] = "type"
TARGET_COLUMN: Final[str] = "quality"
IQR_MULTIPLIER: Final[float] = 1.5
RANDOM_STATE: Final[int] = 42

WineType = Literal["red", "white"]


# --------------------------------------------------------------------------- #
# Configuration & result models
# --------------------------------------------------------------------------- #
class PCAConfig(BaseModel):
    """Parameters of the PCA step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_components: int = Field(default=2, ge=1)
    random_state: int = RANDOM_STATE


class ClusteringConfig(BaseModel):
    """Parameters of the K-means grid search over the number of clusters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=10, ge=2)
    n_init: int = Field(default=10, ge=1)
    random_state: int = RANDOM_STATE

    @model_validator(mode="after")
    def _check_range(self) -> ClusteringConfig:
        if self.k_min > self.k_max:
            raise ValueError(
                f"k_min ({self.k_min}) must be <= k_max ({self.k_max})"
            )
        return self


class ClusteringResult(BaseModel):
    """Validated output of :func:`perform_clustering`."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    labels: np.ndarray
    silhouette_score: float = Field(ge=-1.0, le=1.0)
    n_clusters: int = Field(ge=2)
    scores_by_k: dict[int, float]


# --------------------------------------------------------------------------- #
# Task 1: data preprocessing
# --------------------------------------------------------------------------- #
def _read_wine_csv(path: Path, wine_type: WineType) -> pd.DataFrame:
    df = pd.read_csv(path, sep=CSV_SEPARATOR)
    df.columns = df.columns.str.strip()
    return df.assign(**{TYPE_COLUMN: wine_type})


@validate_call
def load_data(red_path: FilePath, white_path: FilePath) -> pd.DataFrame:
    """
    Load the red and white wine datasets and merge them into one DataFrame.

    A ``type`` column (``'red'`` / ``'white'``) is added based on the source
    file. Rows with missing values and exact duplicates are dropped.
    """
    red = _read_wine_csv(red_path, "red")
    white = _read_wine_csv(white_path, "white")

    if set(red.columns) != set(white.columns):
        raise ValueError(
            "Red and white datasets have different columns: "
            f"{sorted(set(red.columns) ^ set(white.columns))}"
        )

    data = (
        pd.concat([red, white[red.columns]], ignore_index=True)
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info("Loaded %d rows (%d red, %d white)", len(data), len(red), len(white))
    return data


def remove_outliers(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Remove rows containing outliers in any of ``features`` (IQR method)."""
    feature_list = list(features)
    missing = set(feature_list) - set(df.columns)
    if missing:
        raise KeyError(f"Features not found in DataFrame: {sorted(missing)}")

    q1 = df[feature_list].quantile(0.25)
    q3 = df[feature_list].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    in_bounds = df[feature_list].ge(lower) & df[feature_list].le(upper)
    df_clean = df.loc[in_bounds.all(axis=1)].reset_index(drop=True)

    logger.info("Removed %d outlier rows", len(df) - len(df_clean))
    return df_clean


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical ``type`` column (if present)."""
    if TYPE_COLUMN not in data.columns:
        return data.copy()
    return pd.get_dummies(data, columns=[TYPE_COLUMN], prefix=TYPE_COLUMN, dtype=int)


def scale_features(X_train: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Standardize features to zero mean and unit variance."""
    return StandardScaler().fit_transform(X_train)


# --------------------------------------------------------------------------- #
# Task 2: unsupervised learning
# --------------------------------------------------------------------------- #
def apply_pca(
    X: pd.DataFrame | np.ndarray, config: PCAConfig | None = None
) -> np.ndarray:
    """Scale the features and project them onto the principal components."""
    config = config or PCAConfig()
    pca = PCA(n_components=config.n_components, random_state=config.random_state)
    X_pca = pca.fit_transform(scale_features(X))

    logger.info(
        "PCA explained variance ratio: %s (total %.3f)",
        np.round(pca.explained_variance_ratio_, 3),
        pca.explained_variance_ratio_.sum(),
    )
    return X_pca


def find_best_clustering(
    X_pca: np.ndarray, config: ClusteringConfig | None = None
) -> ClusteringResult:
    """Grid-search the number of K-means clusters by silhouette score."""
    config = config or ClusteringConfig()

    best: tuple[int, float, np.ndarray] | None = None
    scores: dict[int, float] = {}
    for k in range(config.k_min, config.k_max + 1):
        labels = KMeans(
            n_clusters=k, n_init=config.n_init, random_state=config.random_state
        ).fit_predict(X_pca)
        score = float(silhouette_score(X_pca, labels))
        scores[k] = score
        logger.debug("k=%d silhouette=%.4f", k, score)
        if best is None or score > best[1]:
            best = (k, score, labels)

    assert best is not None  # guaranteed by ClusteringConfig validation
    k, score, labels = best
    return ClusteringResult(
        labels=labels, silhouette_score=score, n_clusters=k, scores_by_k=scores
    )


def perform_clustering(
    X_pca: np.ndarray, config: ClusteringConfig | None = None
) -> dict[str, float | int | np.ndarray | dict[int, float]]:
    """
    Perform K-means clustering on PCA results.

    Returns a dict with ``labels``, ``silhouette_score``, ``n_clusters`` and
    ``scores_by_k``.
    """
    return find_best_clustering(X_pca, config).model_dump()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def plot_clusters(X_pca: np.ndarray, labels: np.ndarray) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis")
    plt.title("K-means Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data = load_data(
        PROJECT_ROOT / "winequality-red.csv", PROJECT_ROOT / "winequality-white.csv"
    )

    features = data.columns.drop([TARGET_COLUMN, TYPE_COLUMN])
    data = remove_outliers(data, features)
    data = encode_categorical_variables(data)

    X_features = data.drop(columns=[TARGET_COLUMN])
    X_pca = apply_pca(X_features)
    result = find_best_clustering(X_pca)

    print("\nClustering Results:")
    print(f"Optimal number of clusters: {result.n_clusters}")
    print(f"Silhouette coefficient: {result.silhouette_score:.4f}")

    plot_clusters(X_pca, result.labels)


if __name__ == "__main__":
    main()
