import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os, json, logging
import requests
from flask import Blueprint, render_template, redirect, url_for

# cache semplice in memoria
CACHE = {}  # key -> (expires_ts, data)
CACHE_TTL = int(os.environ.get("AIRNOW_CACHE_TTL", "180"))  # default 180s

def _cache_get(key):
    item = CACHE.get(key)
    if not item:
        return None
    exp, data = item
    if exp < time.time():
        CACHE.pop(key, None)
        return None
    return data

def _cache_set(key, data, ttl=CACHE_TTL):
    CACHE[key] = (time.time() + ttl, data)

def _quantize(val, q):  # riduce la frammentazione delle chiavi cache
    return round(float(val) / q) * q

def _quantize_bbox_str(bbox_str, q=0.25):
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox_str.split(",")]
    return f"{_quantize(min_lon,q)},{_quantize(min_lat,q)},{_quantize(max_lon,q)},{_quantize(max_lat,q)}"

# sessione requests con retry/backoff
def _make_session():
    s = requests.Session()
    retries = Retry(
        total=3, connect=3, read=3, status=3,
        backoff_factor=0.75,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

SESSION = _make_session()
log = logging.getLogger(__name__)

# --- aggiungi in cima se non li hai già ---
from datetime import datetime, timedelta, timezone
from math import floor, sqrt
from flask import Blueprint, jsonify, request, Response, render_template

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

bp = Blueprint("main", __name__)  # senza url_prefix per avere /map e /data/...

AIRNOW_BASE = "https://www.airnowapi.org"
API_KEY = os.environ.get("AIRNOW_API_KEY")  # oppure metti la tua stringa qui

# ---------- utils ----------
def _need_api_key():
    if not API_KEY:
        return jsonify({
            "error": "AIRNOW_API_KEY non configurata",
            "how_to_fix": "export AIRNOW_API_KEY=... oppure usa python-dotenv e load_dotenv()"
        }), 500
    return None

def _cell_polygon(lon_min, lat_min, step):
    lon_max = lon_min + step
    lat_max = lat_min + step
    return [
        [lon_min, lat_min],
        [lon_min, lat_max],
        [lon_max, lat_max],
        [lon_max, lat_min],
        [lon_min, lat_min],
    ]

def _clamp_aqi(v):
    try:
        x = float(v)
    except Exception:
        return None
    return max(0.0, min(500.0, x))

# ---------- NUOVO: fetch con /aq/data e BBOX ----------
def _airnow_fetch_bbox(bbox_str, start_date_utc, end_date_utc,
                       parameters="PM25,PM10,O3,CO,SO2,NO2", data_type="A",
                       use_cache=True, cache_quant=0.25):
    url = f"{AIRNOW_BASE}/aq/data/"
    bbox_q = _quantize_bbox_str(bbox_str, q=cache_quant)  # quantizza per riuso cache
    cache_key = ("aqdata", bbox_q, start_date_utc, end_date_utc, parameters, data_type)

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    params = {
        "startDate": start_date_utc, "endDate": end_date_utc,
        "parameters": parameters, "BBOX": bbox_q,
        "dataType": data_type, "format": "application/json",
        "API_KEY": API_KEY
    }
    try:
        r = SESSION.get(url, params=params, timeout=(5, 25))
        r.raise_for_status()
        data = r.json()
        if use_cache:
            _cache_set(cache_key, data)
        return data
    except requests.Timeout as e:
        log.warning("Timeout /aq/data bbox=%s: %s", bbox_q, e); return []
    except requests.HTTPError as e:
        body = getattr(e.response, "text", "") if hasattr(e, "response") else ""
        log.error("HTTPError /aq/data %s; body=%s", e, body[:300]); return []
    except Exception as e:
        log.exception("Errore /aq/data: %s", e); return []


def _split_bbox_4(bbox_str):
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox_str.split(",")]
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0
    return [
        f"{min_lon},{min_lat},{mid_lon},{mid_lat}",
        f"{mid_lon},{min_lat},{max_lon},{mid_lat}",
        f"{min_lon},{mid_lat},{mid_lon},{max_lat}",
        f"{mid_lon},{mid_lat},{max_lon},{max_lat}",
    ]
# ---------- /map (usa il tuo templates/map.html) ----------
@bp.route("/")
def root():
    return redirect(url_for("main.map_page"))

@bp.route("/map")
def map_page():
    return render_template("map.html")

# ---------- /data/aqi_na_grid.geojson (usa /aq/data con BBOX) ----------
def _adaptive_step(min_lon, min_lat, max_lon, max_lat, target_cells=900, min_step=0.25, max_step=1.5):
    """Sceglie uno step in gradi per avere ~target_cells celle nel bbox."""
    width = max(1e-6, max_lon - min_lon)
    height = max(1e-6, max_lat - min_lat)
    # numero di colonne/righe target mantenendo l'aspect ratio
    cols = max(1, int(round(sqrt(target_cells * (width / height)))))
    rows = max(1, int(round(target_cells / cols)))
    step_w = width / cols
    step_h = height / rows
    step = max(step_w, step_h)
    return float(max(min_step, min(step, max_step)))

@bp.route("/data/aqi_na_grid.geojson")
def aqi_na_grid():
    """
    GeoJSON a griglia usando UNICA chiamata AirNow /aq/data filtrata per BBOX.
    - step: se assente o auto_step=1, calcoliamo noi uno step adattivo per non superare ~target_cells.
    - target_cells: default 900 (≈ 30x30); aumenta se vuoi più dettaglio a bassi zoom.
    - parameters: default 'PM25,O3' per ridurre payload (puoi passare 'PM25,PM10,O3,CO,SO2,NO2').
    - dataType: 'A' (aggregato orario) di default.
    - days_ago o startDate/endDate come prima.
    """
    key_err = _need_api_key()
    if key_err:
        return key_err

    bbox_str = request.args.get("bbox")
    if not bbox_str:
        return jsonify({"error": "Parametro 'bbox' obbligatorio (minLon,minLat,maxLon,maxLat)"}), 400

    try:
        min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox_str.split(",")]
    except Exception:
        return jsonify({"error": "bbox non valido. Usa: minLon,minLat,maxLon,maxLat"}), 400

    # parametri
    # Se il client NON passa step, o chiede auto_step=1, usiamo adattivo
    auto_step = request.args.get("auto_step", "1") in ("1", "true", "True")
    target_cells = int(request.args.get("target_cells", 900))
    step_param = request.args.get("step", None)
    min_aqi = float(request.args.get("min_aqi", 51))  # mostra solo dal giallo in su
    if auto_step or step_param is None:
        step = _adaptive_step(min_lon, min_lat, max_lon, max_lat, target_cells=target_cells)
    else:
        step = float(step_param)

    agg = request.args.get("agg", "max")
    parameters = request.args.get("parameters", "PM25,O3")  # ridotto per velocità
    data_type = request.args.get("dataType", "A")

    # --- tempo adattivo se l'utente NON forza start/end ---
    startDate = request.args.get("startDate")
    endDate = request.args.get("endDate")

    width = max_lon - min_lon
    height = max_lat - min_lat
    area_deg2 = max(1e-6, width * height)

    if not (startDate and endDate):
        # area grande => chiedi meno ore
        if area_deg2 >= 600:       # tutto USA/Canada
            hours = int(request.args.get("hours", 1))
        elif area_deg2 >= 200:     # macro regione
            hours = int(request.args.get("hours", 3))
        elif area_deg2 >= 60:      # stato/metro area grande
            hours = int(request.args.get("hours", 6))
        else:
            hours = int(request.args.get("hours", 24))
        now = datetime.utcnow()
        startDate = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H")
        endDate   = now.strftime("%Y-%m-%dT%H")

    # parametri più leggeri quando area enorme
    parameters = request.args.get("parameters") or ("PM25" if area_deg2 >= 600 else "PM25,O3")

    # 1) Fetch punti nel bbox (UNA chiamata)
    observations = _airnow_fetch_bbox(
    bbox_str, startDate, endDate,
    parameters=parameters, data_type=data_type
    )

    # --- fallback: se niente risultati e bbox molto grande, suddividi in 4 e unisci ---
    if not observations and area_deg2 >= 600:
        obs_all = []
        for sub in _split_bbox_4(bbox_str):
            sub_obs = _airnow_fetch_bbox(sub, startDate, endDate, parameters=parameters, data_type=data_type)
            if sub_obs:
                obs_all.extend(sub_obs)
        observations = obs_all


    # 2) Indicizza per cella
    from collections import defaultdict
    cell_values = defaultdict(list)

    width = max_lon - min_lon
    height = max_lat - min_lat
    lon_steps = max(1, int(floor(width / step)))
    lat_steps = max(1, int(floor(height / step)))

    def _cell_origin(lon, lat):
        i_lon = int((lon - min_lon) // step)
        i_lat = int((lat - min_lat) // step)
        i_lon = min(max(i_lon, 0), lon_steps - 1)
        i_lat = min(max(i_lat, 0), lat_steps - 1)
        return (min_lon + i_lon * step, min_lat + i_lat * step)

    for obs in observations or []:
        lat = obs.get("Latitude"); lon = obs.get("Longitude")
        aqi = _clamp_aqi(obs.get("AQI"))
        if lat is None or lon is None or aqi is None:
            continue
        if aqi < min_aqi:
            continue  # ⬅️ niente verdi
        lon0, lat0 = _cell_origin(float(lon), float(lat))
        cell_values[(lon0, lat0)].append(aqi)

    # 3) Costruisci feature (salta celle senza valori -> niente NaN su mappa)
    features = []
    for i_lat in range(lat_steps):
        lat0 = min_lat + i_lat * step
        for i_lon in range(lon_steps):
            lon0 = min_lon + i_lon * step
            vals = cell_values.get((lon0, lat0))
            if not vals:
                continue  # ⬅️ niente feature se non c'è AQI → più leggero
            aqi_val = max(vals) if agg == "max" else (sum(vals)/len(vals))
            if aqi_val < min_aqi:
                continue  # ⬅️ niente feature “verdi”
            poly = _cell_polygon(lon0, lat0, step)
            features.append({
                "type": "Feature",
                "properties": {"AQI": aqi_val},
                "geometry": {"type": "Polygon", "coordinates": [poly]}
            })

    fc = {
        "type": "FeatureCollection",
        "name": "aqi_na_grid",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
        "metadata": {
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "auto_step": auto_step,
            "computed_step": step,
            "target_cells": target_cells,
            "agg": agg,
            "parameters": parameters,
            "dataType": data_type,
            "startDate": startDate,
            "endDate": endDate,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cells": len(features),
            "points_in_bbox": len(observations or [])
        }
    }
    resp = Response(json.dumps(fc), mimetype="application/geo+json")
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp