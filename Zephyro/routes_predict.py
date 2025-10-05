"""Flask blueprint exposing AQI forecast endpoint."""
from __future__ import annotations

import json
import logging
from flask import Blueprint, current_app, jsonify, request, Response

from aqi.infer import predict_week_geojson

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict_aqi", __name__)


def _validate_bbox(bbox: str) -> str:
    parts = bbox.split(",")
    if len(parts) != 4:
        raise ValueError("bbox must contain four comma separated values")
    try:
        _ = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("bbox values must be numeric") from exc
    return bbox


@predict_bp.route("/predict/aqi_week.geojson")
def predict_aqi_week() -> Response:
    bbox = request.args.get("bbox")
    if not bbox:
        return jsonify({"error": "Missing bbox parameter"}), 400
    try:
        bbox = _validate_bbox(bbox)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    auto_step = request.args.get("auto_step", "1") not in {"0", "false", "False"}
    target_cells = int(request.args.get("target_cells", 1200))
    min_aqi = int(request.args.get("min_aqi", 51))
    history_days = int(request.args.get("history_days", 60))

    try:
        geojson = predict_week_geojson(
            bbox=bbox,
            auto_step=auto_step,
            target_cells=target_cells,
            min_aqi=min_aqi,
            history_days=history_days,
            parameters="PM25,O3,NO2"
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("AQI prediction failed: %s", exc)
        return jsonify({"error": "prediction_failed", "details": str(exc)}), 500

    response = current_app.response_class(
        response=json.dumps(geojson),
        status=200,
        mimetype="application/geo+json",
    )
    response.headers["Cache-Control"] = "public, max-age=60"
    return response


__all__ = ["predict_bp"]
