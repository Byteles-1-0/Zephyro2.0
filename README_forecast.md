# AQI Forecasting Module

Questo modulo aggiunge alla codebase la capacità di scaricare, preprocessare, addestrare e servire previsioni AQI giornaliere per 7 giorni.

## Requisiti

* Python 3.10+
* Dipendenze principali: `torch`, `numpy`, `pandas`, `scikit-learn`, `requests`, `Flask`, `python-dotenv`, `Flask-Compress`.

Installa le dipendenze aggiuntive (separatamente dal progetto principale):

```bash
pip install -r Zephyro/requirements.txt
```

## Training

1. Esporta la chiave AirNow:

```bash
export AIRNOW_API_KEY=la_tua_chiave
```

2. Avvia il training (esempio su US continental):

```bash
python -m aqi.train --bbox "-125,24,-66,49" --days_history 180 --parameters "PM25,O3" --out_dir artifacts/
```

Alla fine otterrai:

```
artifacts/
  ├── config.json
  ├── model.pt
  ├── preprocess_config.json
  └── scalers.pkl
```

Il campo `use_baseline` in `config.json` indica se la baseline (persistence/seasonal) ha battuto il Transformer sul validation set.

## Inferenza

L'endpoint Flask è disponibile dopo aver avviato l'app.

```bash
flask --app Zephyro.app run
```

Richiedi un forecast settimanale:

```
/predict/aqi_week.geojson?bbox=-74.3,40.5,-73.6,40.95&auto_step=1&target_cells=1200&min_aqi=51
```

L'output è un GeoJSON `FeatureCollection` con proprietà `AQI_pred_day1` … `AQI_pred_day7`. Le celle con previsioni <= 50 vengono filtrate.

Se i dati storici sono insufficienti per la cella, la risposta riporta `"model": "baseline"` nelle proprietà e `metadata.use_baseline` a `true`.

## Script di diagnostica rapido

Per tracciare le previsioni di una cella specifica:

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
plt.xlabel("Giorni nel futuro")
plt.ylabel("AQI previsto")
plt.title("AQI previsione 7 giorni")
plt.show()
```

## Testing

Esegui i test unitari con:

```bash
pytest
```
