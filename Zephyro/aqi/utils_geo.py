"""Geospatial helper utilities for AQI forecasting."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def _adaptive_step(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    target_cells: int = 1200,
    min_step: float = 0.25,
    max_step: float = 2.0,
) -> float:
    """Compute an adaptive grid step size for the bbox."""
    width = max_lon - min_lon
    height = max_lat - min_lat
    area = max(width * height, 1e-6)
    approx_step = math.sqrt(area / max(target_cells, 1))
    return float(min(max(approx_step, min_step), max_step))


def build_cell_polygon(lon: float, lat: float, step: float) -> List[Tuple[float, float]]:
    """Return a simple square polygon for a grid cell."""
    half = step / 2
    return [
        (lon - half, lat - half),
        (lon + half, lat - half),
        (lon + half, lat + half),
        (lon - half, lat + half),
        (lon - half, lat - half),
    ]


AQI_CATEGORIES: Dict[str, Tuple[int, int]] = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "USG": (101, 150),
    "Unhealthy": (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous": (301, 500),
}


def aqi_category(aqi_value: float) -> str:
    for name, (lo, hi) in AQI_CATEGORIES.items():
        if lo <= aqi_value <= hi:
            return name
    return "Unknown"


__all__ = ["_adaptive_step", "build_cell_polygon", "aqi_category", "AQI_CATEGORIES"]
