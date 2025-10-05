"""Preprocessing utilities for AQI forecasting."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGET_PARAMETER = "PM25"


@dataclass
class DatasetTensors:
    """Container for dataset tensors."""

    X: np.ndarray
    y: np.ndarray


@dataclass
class PreprocessArtifacts:
    """Artifacts produced during preprocessing."""

    scalers: Dict[str, StandardScaler]
    feature_columns: List[str]
    target_columns: List[str]
    config_path: Path


def aggregate_to_daily(df: pd.DataFrame, target_parameter: str = TARGET_PARAMETER) -> pd.DataFrame:
    """Aggregate hourly AQI observations into a daily dataset."""
    if df.empty:
        return pd.DataFrame(columns=["Latitude", "Longitude", "Date", "Parameter", "AQI"])

    working_df = df.copy()
    if "UTC" not in working_df.columns:
        raise ValueError("Input DataFrame must contain 'UTC' column")

    working_df["UTC"] = pd.to_datetime(working_df["UTC"], utc=True)
    working_df["Date"] = working_df["UTC"].dt.floor("D")

    working_df["Parameter"] = working_df["Parameter"].astype(str).str.upper()

    group_cols = ["Latitude", "Longitude", "Date", "Parameter"]
    daily = (
        working_df.groupby(group_cols)["AQI"].max().reset_index()
    )
    if target_parameter and target_parameter not in daily["Parameter"].unique():
        logger.warning("Target parameter %s not found in dataset", target_parameter)

    return daily


def _add_lag_features(df: pd.DataFrame, max_lag: int, value_col: str) -> pd.DataFrame:
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df[value_col].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame, windows: Sequence[int], value_col: str) -> pd.DataFrame:
    for window in windows:
        df[f"roll_mean_{window}"] = df[value_col].rolling(window).mean()
        df[f"roll_max_{window}"] = df[value_col].rolling(window).max()
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dayofweek"] = df["Date"].dt.dayofweek
    dow_encoded = pd.get_dummies(df["dayofweek"], prefix="dow")
    expected_cols = [f"dow_{i}" for i in range(7)]
    dow_encoded = dow_encoded.reindex(columns=expected_cols, fill_value=0)
    df = pd.concat([df, dow_encoded], axis=1)

    day_of_year = df["Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    return df


def _normalize_geo(df: pd.DataFrame) -> pd.DataFrame:
    df["lat_norm"] = (df["Latitude"] - df["Latitude"].mean()) / (df["Latitude"].std() + 1e-6)
    df["lon_norm"] = (df["Longitude"] - df["Longitude"].mean()) / (df["Longitude"].std() + 1e-6)
    return df


def build_features(
    daily_df: pd.DataFrame,
    history_window: int = 60,
    forecast_horizon: int = 7,
    target_parameter: str = TARGET_PARAMETER,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and targets for training."""
    if daily_df.empty:
        raise ValueError("Daily dataframe is empty")
    target_param = target_parameter.replace(" ", "").upper()
    
    target_df = daily_df[daily_df["Parameter"].str.upper() == target_param].copy()
    target_df = target_df.sort_values("Date")

    if target_df.empty:
        raise ValueError(f"No rows for target parameter {target_parameter}")

    target_df = _add_lag_features(target_df, max_lag=14, value_col="AQI")

    target_df = _add_rolling_features(target_df, windows=(3, 7, 14), value_col="AQI")

    target_df = _add_calendar_features(target_df)

    target_df = _normalize_geo(target_df)


    target_df = target_df.dropna().reset_index(drop=True)

    feature_cols = [col for col in target_df.columns if col not in {"AQI", "Parameter", "Date"}]

    X = target_df[feature_cols]
    y = target_df["AQI"]
     
    if len(X) < history_window + forecast_horizon:
        logger.warning("Not enough history to build sequences: len=%s", len(X))

    return X, y


def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    history_window: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create rolling window sequences for seq2seq models."""
    feature_cols = list(X.columns)
    X_values = X.values
    y_values = y.values

    sequences_X: List[np.ndarray] = []
    sequences_y: List[np.ndarray] = []

    for start in range(0, len(X_values) - history_window - forecast_horizon + 1):
        end = start + history_window
        target_end = end + forecast_horizon
        sequences_X.append(X_values[start:end])
        sequences_y.append(y_values[end:target_end])

    if not sequences_X:
        raise ValueError("Unable to create sequences with the provided history window")

    return np.stack(sequences_X), np.stack(sequences_y), feature_cols


def fit_scalers(X: np.ndarray, y: np.ndarray) -> Dict[str, StandardScaler]:
    scalers = {
        "X": StandardScaler(),
        "y": StandardScaler(),
    }
    b, t, f = X.shape
    X_2d = X.reshape(b * t, f)
    scalers["X"].fit(X_2d)
    scalers["y"].fit(y.reshape(-1, 1))
    return scalers


def transform_with_scalers(X: np.ndarray, y: np.ndarray, scalers: Dict[str, StandardScaler]) -> Tuple[np.ndarray, np.ndarray]:
    b, t, f = X.shape
    X_scaled = scalers["X"].transform(X.reshape(b * t, f)).reshape(b, t, f)
    y_scaled = scalers["y"].transform(y.reshape(-1, 1)).reshape(y.shape)
    return X_scaled, y_scaled


def inverse_transform_targets(y_scaled: np.ndarray, scalers: Dict[str, StandardScaler]) -> np.ndarray:
    return scalers["y"].inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)


def save_preprocess_artifacts(artifacts: PreprocessArtifacts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scalers_path = out_dir / "scalers.pkl"
    with scalers_path.open("wb") as f:
        import pickle

        pickle.dump(artifacts.scalers, f)

    config_path = out_dir / "preprocess_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({
            "feature_columns": artifacts.feature_columns,
            "target_columns": artifacts.target_columns,
        }, f, indent=2)


def load_scalers(path: Path) -> Dict[str, StandardScaler]:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


__all__ = [
    "aggregate_to_daily",
    "build_features",
    "create_sequences",
    "fit_scalers",
    "transform_with_scalers",
    "inverse_transform_targets",
    "DatasetTensors",
    "PreprocessArtifacts",
    "save_preprocess_artifacts",
    "load_scalers",
]
