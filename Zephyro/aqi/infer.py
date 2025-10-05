"""Inference utilities for AQI weekly forecasts."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from .data_ingest import fetch_airnow_bbox
from .preprocess import (
    aggregate_to_daily,
    build_features,
    create_sequences,
    load_scalers,
    transform_with_scalers,
    inverse_transform_targets,
)
from .model_transformer import AQITransformer, ModelConfig, predict
from .baselines import persistence_7d, seasonal_naive_7d
from .utils_geo import _adaptive_step, build_cell_polygon

logger = logging.getLogger(__name__)


class ForecastArtifacts:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        with (artifacts_dir / "config.json").open("r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.scalers = load_scalers(artifacts_dir / "scalers.pkl")
        self.model_state = torch.load(artifacts_dir / "model.pt", map_location="cpu")
        preprocess_cfg_path = artifacts_dir / "preprocess_config.json"
        if preprocess_cfg_path.exists():
            with preprocess_cfg_path.open("r", encoding="utf-8") as f:
                self.preprocess_config = json.load(f)
        else:
            self.preprocess_config = {"feature_columns": []}

    def build_model(self) -> AQITransformer:
        input_size = self.model_state["input_projection.weight"].shape[1]
        config = ModelConfig(input_size=input_size, forecast_horizon=self.config.get("forecast_horizon", 7))
        model = AQITransformer(config)
        model.load_state_dict(self.model_state)
        return model


def _prepare_history(bbox: str, history_days: int, parameters: str) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=history_days)

    return fetch_airnow_bbox(
        bbox=bbox,
        start_utc="2025-07-07T09",#start_dt.strftime("%Y-%m-%dT%H"),
        end_utc="2025-10-07T09",#end_dt.strftime("%Y-%m-%dT%H"),
        parameters=parameters,
    )


def predict_week_geojson(
    bbox: str,
    auto_step: bool = True,
    target_cells: int = 1200,
    min_aqi: int = 51,
    parameters: str = "PM25,O3",
    history_days: int = 60,
    artifacts_dir: str | Path = "artifacts",
) -> Dict[str, object]:
    """Generate weekly AQI predictions as GeoJSON."""
    min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    step = _adaptive_step(min_lon, min_lat, max_lon, max_lat, target_cells=target_cells) if auto_step else 0.5

    artifacts = ForecastArtifacts(Path(artifacts_dir))
    model = artifacts.build_model()
    use_baseline = artifacts.config.get("use_baseline", False)
    
    history_df = _prepare_history(bbox, history_days=history_days, parameters=parameters)
  
    target_parameter = "PM2.5"
    daily = aggregate_to_daily(history_df, target_parameter)
    if not daily.empty:
        daily["Parameter"] = daily["Parameter"].astype(str).str.upper()

    features_list = []
    default_model_name = "baseline" if use_baseline else "transformer"
    model_name = default_model_name

    grid_lons = np.arange(min_lon + step / 2, max_lon, step)
    grid_lats = np.arange(min_lat + step / 2, max_lat, step)
    for lon in grid_lons:
        for lat in grid_lats:
            half_step = step / 2
            lon_min, lon_max = lon - half_step, lon + half_step
            lat_min, lat_max = lat - half_step, lat + half_step
            model_name = default_model_name
            #print(daily)
            if daily.empty:
                continue
            # print(lon_min,lon_max,lat_min,lat_max)
            # print(daily)
            cell_mask = (
                (daily["Longitude"] >= lon_min)
                & (daily["Longitude"] < lon_max)
                & (daily["Latitude"] >= lat_min)
                & (daily["Latitude"] < lat_max)
                & (daily["Parameter"] == "PM2.5")
            )
            parameter_mask = daily["Parameter"] == target_parameter
            cell_obs = daily[cell_mask & parameter_mask]
            if cell_obs.empty:
                continue

            series = (
                cell_obs.groupby("Date", as_index=False)["AQI"].mean().sort_values("Date")
            )
            series["Parameter"] = target_parameter
            series["Latitude"] = lat
            series["Longitude"] = lon
            series = series[["Latitude", "Longitude", "Date", "Parameter", "AQI"]]
            try:
                cell_features, cell_target = build_features(
                    series,
                    history_window=artifacts.config.get("history_window", 45),
                    target_parameter=target_parameter,
                )
            except ValueError:
                cell_features = pd.DataFrame()
                cell_target = pd.Series(dtype=float)
            preds_use: np.ndarray | None
            if cell_features.empty or len(cell_features) < artifacts.config.get("history_window", 45):
                preds_use = seasonal_naive_7d(series["AQI"].values)
                model_name = "baseline"
            else:
                feature_cols = artifacts.preprocess_config.get("feature_columns", list(cell_features.columns))
                for col in feature_cols:
                    if col not in cell_features.columns:
                        cell_features[col] = 0.0
                cell_features = cell_features.reindex(columns=feature_cols, fill_value=0)
                try:
                    X_cell, y_cell, _ = create_sequences(
                        cell_features,
                        cell_target,
                        history_window=artifacts.config.get("history_window", 45),
                        forecast_horizon=artifacts.config.get("forecast_horizon", 7),
                    )
                except ValueError:
                    preds_use = seasonal_naive_7d(series["AQI"].values)
                    model_name = "baseline"
                else:
                    scalers = artifacts.scalers
                    X_scaled, _ = transform_with_scalers(X_cell, y_cell, scalers)
                    X_tensor = torch.tensor(X_scaled[-1:], dtype=torch.float32)
                    if use_baseline:
                        preds_use = seasonal_naive_7d(series["AQI"].values)
                        model_name = "baseline"
                    else:
                        preds_scaled = predict(model, X_tensor)
                        preds_use = inverse_transform_targets(preds_scaled.numpy(), scalers)[0]
                        model_name = "transformer"

                if preds_use is None or np.max(preds_use) < min_aqi:
                    continue

                properties = {f"AQI_pred_day{i+1}": float(preds_use[i]) for i in range(len(preds_use))}
                properties.update({"model": model_name, "lat": float(lat), "lon": float(lon)})

                polygon = build_cell_polygon(lon, lat, step)
                features_list.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[list(coord) for coord in polygon]]},
                        "properties": properties,
                    }
                )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bbox": bbox,
        "step": step,
        "model_name": model_name if features_list else default_model_name,
        "use_baseline": use_baseline,
    }

    return {"type": "FeatureCollection", "features": features_list, "metadata": metadata}


__all__ = ["predict_week_geojson"]
