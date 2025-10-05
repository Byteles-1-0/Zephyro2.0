from pathlib import Path

import pandas as pd

from aqi import data_ingest


def test_fetch_airnow_bbox_tiles_large_bbox(monkeypatch, tmp_path):
    calls = []

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(data_ingest, "CACHE_DIR", cache_dir)

    def fake_to_parquet(self, path, *args, **kwargs):  # noqa: D401 - simple stub
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    def fake_fetch_airnow_tile(request, session, api_key):
        calls.append(request.bbox)
        return pd.DataFrame(
            [
                {
                    "Latitude": 40.0,
                    "Longitude": -75.0,
                    "UTC": "2024-01-01T00",
                    "Parameter": "PM25",
                    "AQI": 55,
                }
            ]
        )

    monkeypatch.setattr(data_ingest, "_fetch_airnow_tile", fake_fetch_airnow_tile)
    monkeypatch.setattr(data_ingest, "_build_session", lambda: object())
    monkeypatch.setattr(data_ingest, "AIRNOW_MAX_TILE_SPAN_DEGREES", 5.0)

    df = data_ingest.fetch_airnow_bbox(
        bbox="-125,24,-66,49",
        start_utc="2024-01-01T00",
        end_utc="2024-01-02T00",
        parameters="PM25",
        api_key="dummy",
    )

    assert len(calls) > 1, "Large bbox should be split into multiple tiles"
    assert not df.empty
    assert set(["Latitude", "Longitude", "UTC", "Parameter", "AQI"]).issubset(df.columns)