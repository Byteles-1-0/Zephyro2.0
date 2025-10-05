"""Training CLI for AQI forecasting Transformer."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .data_ingest import fetch_airnow_bbox
from .preprocess import (
    aggregate_to_daily,
    build_features,
    create_sequences,
    fit_scalers,
    transform_with_scalers,
    save_preprocess_artifacts,
    PreprocessArtifacts,
)
from .model_transformer import AQITransformer, ModelConfig, train_model
from .baselines import mae, smape, persistence_7d, seasonal_naive_7d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _time_range(days_history: int) -> Tuple[str, str]:
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_history)
    #return start_dt.strftime("%Y-%m-%dT%H"), end_dt.strftime("%Y-%m-%dT%H")
    return "2025-07-07T09","2025-10-07T09"


def train_pipeline(args: argparse.Namespace) -> Dict[str, str]:
    start_utc, end_utc = _time_range(args.days_history)
    raw_df = fetch_airnow_bbox(args.bbox, start_utc, end_utc, parameters=args.parameters)
    daily = aggregate_to_daily(raw_df,"PM2.5")
    X_df, y_series = build_features(daily, history_window=args.history_window,target_parameter="PM2.5")

    X, y, feature_cols = create_sequences(
        X_df,
        y_series,
        history_window=args.history_window,
        forecast_horizon=args.forecast_horizon,
    )

    split_idx = max(1, len(X) - args.val_days)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(X_val) == 0:
        X_val = X_train[-1:]
        y_val = y_train[-1:]

    scalers = fit_scalers(X_train, y_train)
    X_train_s, y_train_s = transform_with_scalers(X_train, y_train, scalers)
    X_val_s, y_val_s = transform_with_scalers(X_val, y_val, scalers)

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    config = ModelConfig(input_size=X_train.shape[-1], forecast_horizon=args.forecast_horizon)
    model = AQITransformer(config)

    train_tensors = (torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train_s, dtype=torch.float32))
    val_tensors = (torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val_s, dtype=torch.float32))

    train_result = train_model(model, train_tensors, val_tensors, epochs=args.epochs, lr=args.lr, device=device)

    with torch.no_grad():
        val_preds_scaled = model(val_tensors[0].to(device)).cpu().numpy()
    val_preds = scalers["y"].inverse_transform(val_preds_scaled.reshape(-1, 1)).reshape(val_preds_scaled.shape)

    y_val_true = y_val
    baseline_persistence = np.stack([persistence_7d(seq[-1]) for seq in y_val_true])
    baseline_seasonal = np.stack([seasonal_naive_7d(seq) for seq in y_val_true])

    transformer_mae = mae(y_val_true.reshape(-1), val_preds.reshape(-1))
    persistence_mae = mae(y_val_true.reshape(-1), baseline_persistence.reshape(-1))
    seasonal_mae = mae(y_val_true.reshape(-1), baseline_seasonal.reshape(-1))

    use_baseline = transformer_mae > min(persistence_mae, seasonal_mae) * (1 - args.baseline_margin)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    artifacts = PreprocessArtifacts(
        scalers=scalers,
        feature_columns=feature_cols,
        target_columns=["AQI"],
        config_path=out_dir / "preprocess_config.json",
    )
    save_preprocess_artifacts(artifacts, out_dir)

    config_dict = {
        "bbox": args.bbox,
        "history_window": args.history_window,
        "forecast_horizon": args.forecast_horizon,
        "val_days": args.val_days,
        "epochs": args.epochs,
        "lr": args.lr,
        "use_baseline": use_baseline,
        "metrics": {
            "transformer_mae": transformer_mae,
            "persistence_mae": persistence_mae,
            "seasonal_mae": seasonal_mae,
        },
    }

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training completed. Artifacts saved to %s", out_dir)
    return {"model_path": str(model_path), "config_path": str(out_dir / "config.json")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AQI Transformer model")
    parser.add_argument("--bbox", help="Bounding box minLon,minLat,maxLon,maxLat", default="-74.259090,40.477399,-73.700272,40.917577")
    parser.add_argument("--days_history", type=int, default=180)
    parser.add_argument("--parameters", default="PM25,O3,NO2")
    parser.add_argument("--forecast_horizon", type=int, default=7)
    parser.add_argument("--history_window", type=int, default=10)
    parser.add_argument("--val_days", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--baseline_margin", type=float, default=0.05)
    parser.add_argument("--out_dir", default="artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_pipeline(args)


if __name__ == "__main__":
    main()
