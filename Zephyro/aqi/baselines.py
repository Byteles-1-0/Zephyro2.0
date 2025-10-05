"""Baseline models for AQI forecasting."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def persistence_7d(last_value: float) -> np.ndarray:
    """Return persistence baseline predictions."""
    return np.full(7, float(last_value))


def seasonal_naive_7d(series: Iterable[float]) -> np.ndarray:
    """Use the values from the previous week when available."""
    values = list(series)
    if len(values) < 7:
        return persistence_7d(values[-1] if values else 0.0)
    return np.array(values[-7:])


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)))


__all__ = ["persistence_7d", "seasonal_naive_7d", "mae", "smape"]
