import json
from datetime import datetime, timedelta, timezone

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
sk_pre = pytest.importorskip("sklearn.preprocessing")
StandardScaler = sk_pre.StandardScaler

from aqi import infer
from aqi import model_transformer


def _make_artifacts(tmp_path):
    feature_cols = ["Latitude", "Longitude", "lag_1", "roll_mean_3", "dow_0"]
    scalers = {
        "X": StandardScaler().fit(np.zeros((10, len(feature_cols))))
    }
    scalers["y"] = StandardScaler().fit(np.zeros((10, 1)))

    with (tmp_path / "scalers.pkl").open("wb") as f:
        import pickle

        pickle.dump(scalers, f)

    config = {
        "history_window": 5,
        "forecast_horizon": 7,
        "use_baseline": True,
    }
    with (tmp_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f)

    with (tmp_path / "preprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f)

    model = model_transformer.AQITransformer(model_transformer.ModelConfig(input_size=len(feature_cols)))
    torch.save(model.state_dict(), tmp_path / "model.pt")


def test_predict_week_geojson_baseline(monkeypatch, tmp_path):
    _make_artifacts(tmp_path)

    def fake_fetch(*args, **kwargs):
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows = []
        for i in range(10):
            dt = end - timedelta(days=10 - i)
            rows.append({
                "Latitude": 40.0,
                "Longitude": -75.0,
                "UTC": dt.isoformat().replace("+00:00", ""),
                "Parameter": "PM25",
                "AQI": 80 + i,
            })
        return pd.DataFrame(rows)

    monkeypatch.setattr(infer, "fetch_airnow_bbox", fake_fetch)

    result = infer.predict_week_geojson(
        bbox="-75.5,39.5,-74.5,40.5",
        artifacts_dir=tmp_path,
        history_days=5,
        min_aqi=51,
    )

    assert result["type"] == "FeatureCollection"
    assert result["features"]
    for feature in result["features"]:
        assert feature["properties"]["model"] == "baseline"
        assert len([k for k in feature["properties"] if k.startswith("AQI_pred_day")]) == 7

def test_predict_week_geojson_cell_aggregation(monkeypatch, tmp_path):
    _make_artifacts(tmp_path)

    def fake_fetch(*args, **kwargs):
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows = []
        coordinates = [
            (39.6, -75.3),
            (40.3, -74.7),
        ]
        for lat, lon in coordinates:
            for i in range(10):
                dt = end - timedelta(days=10 - i)
                rows.append(
                    {
                        "Latitude": lat,
                        "Longitude": lon,
                        "UTC": dt.isoformat().replace("+00:00", ""),
                        "Parameter": "PM25",
                        "AQI": 60 + i,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(infer, "fetch_airnow_bbox", fake_fetch)
    #"-74.3,38.5,-76.6,43.95"
    result = infer.predict_week_geojson(
        bbox="-74.3,40.5,-73.6,40.95",
        artifacts_dir=tmp_path,
        history_days=5,
        min_aqi=0,
        auto_step=False,
    )

    assert result["features"]
    centers = {(feat["properties"]["lat"], feat["properties"]["lon"]) for feat in result["features"]}
    expected_centers = {(39.75, -75.25), (40.25, -74.75)}
    assert len(centers) == len(expected_centers)
    for expected_lat, expected_lon in expected_centers:
        assert any(
            pytest.approx(expected_lat, rel=1e-4) == lat
            and pytest.approx(expected_lon, rel=1e-4) == lon
            for lat, lon in centers
        )

def test_predict_week_geojson_distinct_predictions(monkeypatch, tmp_path):
    _make_artifacts(tmp_path)

    def fake_fetch(*args, **kwargs):
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        rows = []
        tile_specs = [
            ((39.6, -75.3), 40),
            ((40.3, -74.7), 70),
        ]
        for (lat, lon), base in tile_specs:
            for offset, days_ago in enumerate(range(13, -1, -1)):
                dt = end - timedelta(days=days_ago)
                rows.append(
                    {
                        "Latitude": lat,
                        "Longitude": lon,
                        "UTC": dt.isoformat().replace("+00:00", ""),
                        "Parameter": "PM25",
                        "AQI": base + offset,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(infer, "fetch_airnow_bbox", fake_fetch)

    result = infer.predict_week_geojson(
        bbox="-75.5,39.5,-74.5,40.5",
        artifacts_dir=tmp_path,
        history_days=14,
        min_aqi=0,
        auto_step=False,
    )

    assert len(result["features"]) == 2
    predictions = {}
    for feature in result["features"]:
        props = feature["properties"]
        key = (round(props["lat"], 2), round(props["lon"], 2))
        predictions[key] = [props[f"AQI_pred_day{i}"] for i in range(1, 8)]

    expected = {
        (39.75, -75.25): [40 + i for i in range(7, 14)],
        (40.25, -74.75): [70 + i for i in range(7, 14)],
    }

    assert predictions.keys() == expected.keys()
    unique_predictions = {tuple(vals) for vals in predictions.values()}
    assert len(unique_predictions) == 2
    for key, preds in predictions.items():
        assert preds == expected[key]