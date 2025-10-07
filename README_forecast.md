# AQI Forecasting Module

This module adds to the codebase the ability to download, preprocess, train, and serve 7-day daily AQI forecasts.

## Requisiti

* Python 3.10+
* Main dependencies: torch, numpy, pandas, scikit-learn, requests, Flask, python-dotenv, Flask-Compress.

Install the additional dependencies (separately from the main project):

```bash
pip install -r Zephyro/requirements.txt
```

## Training

1. Export your AirNow key:

```bash
export AIRNOW_API_KEY=your_key
```

2. Start training (example on continental US):

```bash
python -m aqi.train --bbox "-125,24,-66,49" --days_history 180 --parameters "PM25,O3" --out_dir artifacts/
```

At the end you will get:

```
artifacts/
  ├── config.json
  ├── model.pt
  ├── preprocess_config.json
  └── scalers.pkl
```

The use_baseline field in config.json indicates whether the baseline (persistence/seasonal) outperformed the Transformer on the validation set.

## Inference

The Flask endpoint is available once the app is running:

```bash
flask --app Zephyro.app run
```

Request a weekly forecast:

```
/predict/aqi_week.geojson?bbox=-74.3,40.5,-73.6,40.95&auto_step=1&target_cells=1200&min_aqi=51
```

The output is a GeoJSON FeatureCollection with properties AQI_pred_day1 … AQI_pred_day7.
Cells with forecasts ≤ 50 are filtered out.

If historical data are insufficient for a cell, the response will include "model": "baseline" in the properties and metadata.use_baseline set to true.
## Quick Diagnostic Script

To plot the forecast for a specific cell:

```python
import json
import matplotlib.pyplot as plt
import requests

resp = requests.get("http://localhost:5000/predict/aqi_week.geojson", params={
    "bbox": "-74.3,40.5,-73.6,40.95",
    "auto_step": 1,
    "target_cells": 1200,
    "min_aqi": 51,
})

data = resp.json()
feature = data["features"][0]
values = [feature["properties"][f"AQI_pred_day{i}"] for i in range(1, 8)]
plt.plot(range(1, 8), values, marker="o")
plt.xlabel("Days ahead")
plt.ylabel("Predicted AQI")
plt.title("7-Day AQI Forecast")
plt.show()
```

## Testing

Run the unit tests with:

```bash
pytest
```
