"""Utilities for downloading and caching AQI observations from AirNow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import logging
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import os

# Load environment variables from .env if available
load_dotenv()

logger = logging.getLogger(__name__)

AIRNOW_BASE_URL = "https://www.airnowapi.org/aq/data/"
CACHE_DIR = Path(os.environ.get("AIRNOW_CACHE_DIR", "./.airnow_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# AirNow seems to reject very large bounding boxes. Requests bigger than ~10° per side
# are prone to HTTP 400 errors, therefore we tile the bbox into smaller chunks.
AIRNOW_MAX_TILE_SPAN_DEGREES = 10.0
EXPECTED_COLUMNS = ["Latitude", "Longitude", "UTC", "Parameter", "AQI"]
ISO_HOUR_FORMAT = "%Y-%m-%dT%H"


@dataclass(frozen=True)
class AirNowRequest:
    """Container describing a single AirNow API request."""

    bbox: str
    start_utc: str
    end_utc: str
    parameters: str
    data_type: str

    def cache_key(self) -> str:
        """Create a cache key for the request."""
        payload = json.dumps(
            {
                "bbox": self.bbox,
                "start_utc": self.start_utc,
                "end_utc": self.end_utc,
                "parameters": self.parameters,
                "data_type": self.data_type,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"airnow_{key}.parquet"


def _finalize_and_cache(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    else:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        for col in missing:
            df[col] = pd.NA
        df = df.drop_duplicates(
            subset=["Latitude", "Longitude", "UTC", "Parameter"], keep="last"
        )

    df.to_parquet(cache_path, index=False)
    logger.info("Cached AirNow data to %s", cache_path)
    return df


def _split_bbox_quadrants(bbox: str) -> Iterable[str]:
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox.split(",")]
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0
    return (
        f"{min_lon},{min_lat},{mid_lon},{mid_lat}",
        f"{mid_lon},{min_lat},{max_lon},{mid_lat}",
        f"{min_lon},{mid_lat},{mid_lon},{max_lat}",
        f"{mid_lon},{mid_lat},{max_lon},{max_lat}",
    )


def _tile_bbox(bbox: str, max_span: float) -> Iterable[str]:
    """Split the bbox into tiles that respect the AirNow size constraints."""

    tiles: list[str] = []
    queue: deque[str] = deque([bbox])

    while queue:
        current = queue.popleft()
        min_lon, min_lat, max_lon, max_lat = [float(x) for x in current.split(",")]
        width = max_lon - min_lon
        height = max_lat - min_lat

        if width <= max_span and height <= max_span:
            tiles.append(current)
            continue

        queue.extend(_split_bbox_quadrants(current))

    return tiles


def _fetch_airnow_tile(
    request: AirNowRequest,
    session: requests.Session,
    api_key: str,
) -> pd.DataFrame:
    cache_path = _cache_path(request.cache_key())

    if cache_path.exists():
        logger.info("Loading AirNow data from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    params = {
        "startDate": request.start_utc,
        "endDate": request.end_utc,
        "bbox": request.bbox,
        "parameters": request.parameters.replace(".", ""),
        "datatype": request.data_type,
        "format": "application/json",
        "API_KEY": api_key,
    }

    logger.info(
        "Fetching AirNow data: bbox=%s, start=%s, end=%s",
        request.bbox,
        request.start_utc,
        request.end_utc,
    )

    response = session.get(AIRNOW_BASE_URL, params=params, timeout=60)

    if response.status_code == 400:
        message = response.text
        if "exceeds the record query limit" in message.lower():
            start_dt = datetime.strptime(request.start_utc, ISO_HOUR_FORMAT)
            end_dt = datetime.strptime(request.end_utc, ISO_HOUR_FORMAT)
            if end_dt - start_dt <= timedelta(hours=1):
                raise RuntimeError(
                    "AirNow API request failed even after splitting to 1-hour window"
                )

            midpoint = start_dt + (end_dt - start_dt) / 2
            midpoint = midpoint.replace(minute=0, second=0, microsecond=0)
            if midpoint <= start_dt:
                midpoint = start_dt + timedelta(hours=1)
            if midpoint >= end_dt:
                midpoint = end_dt - timedelta(hours=1)

            if midpoint <= start_dt or midpoint >= end_dt:
                raise RuntimeError(
                    "Unable to split AirNow request further to satisfy record limit"
                )

            midpoint_str = midpoint.strftime(ISO_HOUR_FORMAT)
            logger.info(
                "Splitting AirNow request %s at %s to stay under record limit",
                request,
                midpoint_str,
            )

            first_request = AirNowRequest(
                bbox=request.bbox,
                start_utc=request.start_utc,
                end_utc=midpoint_str,
                parameters=request.parameters,
                data_type=request.data_type,
            )
            second_request = AirNowRequest(
                bbox=request.bbox,
                start_utc=midpoint_str,
                end_utc=request.end_utc,
                parameters=request.parameters,
                data_type=request.data_type,
            )

            frames = [
                df
                for df in (
                    _fetch_airnow_tile(first_request, session=session, api_key=api_key),
                    _fetch_airnow_tile(second_request, session=session, api_key=api_key),
                )
                if not df.empty
            ]

            if frames:
                combined = pd.concat(frames, ignore_index=True)
            else:
                combined = pd.DataFrame(columns=EXPECTED_COLUMNS)

            return _finalize_and_cache(combined, cache_path)

    if response.status_code != 200:
        raise RuntimeError(
            f"AirNow API request failed with status {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError("Unable to decode AirNow API response as JSON") from exc

    if not data:
        logger.warning("AirNow API returned no data for request %s", request)
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    else:
        df = pd.DataFrame(data)

    return _finalize_and_cache(df, cache_path)


def fetch_airnow_bbox(
    bbox: str,
    start_utc: str,
    end_utc: str,
    parameters: str = "PM2.5,O3",
    data_type: str = "A",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch AirNow AQI data for a bounding box.

    Parameters
    ----------
    bbox: str
        Bounding box as "minLon,minLat,maxLon,maxLat".
    start_utc: str
        Start datetime (inclusive) in ISO hour format ("YYYY-MM-DDTHH").
    end_utc: str
        End datetime (exclusive) in ISO hour format.
    parameters: str, optional
        Comma separated pollutant parameters.
    data_type: str, optional
        Data type filter, defaults to "A" (aggregated).
    api_key: str, optional
        Explicit API key. If omitted, the AIRNOW_API_KEY environment variable is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing at least Latitude, Longitude, UTC, Parameter, AQI columns.
    """

    request = AirNowRequest(
        bbox=bbox,
        start_utc=start_utc,
        end_utc=end_utc,
        parameters=parameters,
        data_type=data_type,
    )
    cache_path = _cache_path(request.cache_key())

    if cache_path.exists():
        logger.info("Loading AirNow data from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    key = api_key or os.environ.get("AIRNOW_API_KEY")
    if not key:
        raise RuntimeError(
            "AIRNOW_API_KEY not set. Please export it or provide api_key parameter."
        )

    tiles = list(_tile_bbox(bbox, AIRNOW_MAX_TILE_SPAN_DEGREES))
    if len(tiles) > 1:
        logger.info(
            "Tiling bbox %s into %d requests because it exceeds max span %.1f°",
            bbox,
            len(tiles),
            AIRNOW_MAX_TILE_SPAN_DEGREES,
        )

    session = _build_session()
    frames: list[pd.DataFrame] = []

    for tile_bbox in tiles:
        tile_request = AirNowRequest(
            bbox=tile_bbox,
            start_utc=start_utc,
            end_utc=end_utc,
            parameters=parameters,
            data_type=data_type,
        )
        tile_df = _fetch_airnow_tile(tile_request, session=session, api_key=key)
        if not tile_df.empty:
            frames.append(tile_df)

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["Latitude", "Longitude", "UTC", "Parameter", "AQI"])

    if not df.empty:
        df = df.drop_duplicates(subset=["Latitude", "Longitude", "UTC", "Parameter"], keep="last")

    df.to_parquet(cache_path, index=False)
    logger.info("Cached AirNow data to %s", cache_path)
    return df


__all__ = ["fetch_airnow_bbox", "AirNowRequest"]