import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from aqi.preprocess import aggregate_to_daily, build_features, create_sequences


def make_hourly_df():
    hours = pd.date_range("2023-01-01", periods=48, freq="H", tz="UTC")
    data = []
    for hour in hours:
        data.append({
            "Latitude": 40.0,
            "Longitude": -75.0,
            "UTC": hour,
            "Parameter": "PM25",
            "AQI": int(hour.hour),
        })
    return pd.DataFrame(data)


def test_aggregate_to_daily_max():
    df = make_hourly_df()
    daily = aggregate_to_daily(df)
    assert len(daily) == 2
    assert daily.iloc[0]["AQI"] == 23


def test_build_features_creates_lags():
    df = aggregate_to_daily(make_hourly_df())
    features, target = build_features(df, history_window=10)
    assert "lag_1" in features.columns
    assert len(features) == len(target)


def test_create_sequences_shapes():
    df = aggregate_to_daily(make_hourly_df())
    features, target = build_features(df, history_window=10)
    X, y, cols = create_sequences(features, target, history_window=10, forecast_horizon=3)
    assert X.shape[1] == 10
    assert y.shape[1] == 3
    assert len(cols) == X.shape[2]
