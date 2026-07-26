"""
REST API Routes
================
JSON API endpoints for predictions, segment data, and dashboard charts.
Decoupled from the frontend — consumable by any client.
"""

import json
import pandas as pd
from flask import Blueprint, request, jsonify
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.clv_model import predict_clv, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.clustering import predict_segment
from src.roi_calculator import calculate_roi, get_all_segments_roi

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict CLV for a customer.

    Expects JSON body with customer features:
    {
        "recency": 30,
        "frequency": 3,
        "monetary": 450.0,
        ...
    }

    Returns:
    {
        "predicted_clv": 1234.56,
        "segment_id": 0,
        "segment_name": "Premium Loyalists",
        "confidence": "high"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Set defaults for optional fields
        defaults = {
            "tenure_days": 0,
            "avg_discount": 15,
            "avg_quantity": 2,
            "avg_rating": 3.5,
            "avg_shipping": 12,
            "return_rate": 0.1,
            "total_items": 5,
            "n_returns": 0,
            "category_diversity": 2,
            "avg_product_price": 300,
            "purchase_rate": 0.01,
            "avg_order_value": 400,
            "dominant_category": "Electronics",
            "dominant_channel": "Organic",
            "country": "USA",
            "dominant_payment": "Credit Card",
        }

        for key, default in defaults.items():
            if key not in data:
                data[key] = default

        # Predict CLV
        predicted_clv = predict_clv(data)

        # Predict segment
        segment_id = predict_segment(data)
        profiles = _load_json(config.SEGMENT_PROFILES_PATH)
        profile = profiles.get(str(segment_id), {})

        # Confidence based on frequency
        freq = data.get("frequency", 1)
        confidence = "high" if freq >= 5 else "medium" if freq >= 2 else "low"

        return jsonify({
            "predicted_clv": predicted_clv,
            "segment_id": segment_id,
            "segment_name": profile.get("persona_name", f"Segment {segment_id}"),
            "segment_emoji": profile.get("emoji", "📊"),
            "segment_color": profile.get("color", "#888"),
            "confidence": confidence,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/segments", methods=["GET"])
def get_segments():
    """Get all segment profiles."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    return jsonify(profiles)


@api_bp.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    """
    Get chart data for the dashboard.
    Returns segment distributions, CLV distributions, metrics, etc.
    """
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    metrics = _load_json(config.PROCESSED_DIR / "model_metrics.json")
    shap_data = _load_json(config.PROCESSED_DIR / "shap_analysis.json")

    # Segment distribution for pie chart
    segment_chart = {
        "labels": [],
        "sizes": [],
        "colors": [],
        "avg_clvs": [],
    }
    for seg_id, prof in profiles.items():
        segment_chart["labels"].append(f"{prof['persona_name']}")
        segment_chart["sizes"].append(prof["size"])
        segment_chart["colors"].append(prof["color"])
        segment_chart["avg_clvs"].append(prof["avg_clv"])

    # SHAP feature importance for bar chart
    shap_chart = {"features": [], "importances": []}
    if "global_importance" in shap_data:
        for item in shap_data["global_importance"][:8]:
            shap_chart["features"].append(item["feature"])
            shap_chart["importances"].append(item["importance"])

    return jsonify({
        "segment_chart": segment_chart,
        "metrics": metrics,
        "shap_chart": shap_chart,
    })


@api_bp.route("/roi", methods=["POST"])
def calculate_roi_endpoint():
    """
    Calculate ROI for a retention campaign.

    Expects JSON:
    {
        "segment_id": 0,
        "retention_uplift_pct": 15,
        "months": 12
    }
    """
    try:
        data = request.get_json()
        profiles = _load_json(config.SEGMENT_PROFILES_PATH)

        segment_id = str(data.get("segment_id", 0))
        if segment_id not in profiles:
            return jsonify({"error": f"Unknown segment: {segment_id}"}), 400

        roi = calculate_roi(
            profiles[segment_id],
            data.get("retention_uplift_pct", 15),
            data.get("months", 12),
        )

        return jsonify(roi)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/roi/all", methods=["GET"])
def all_segments_roi():
    """Get ROI estimates for all segments."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    uplift = request.args.get("uplift", 15, type=float)
    months = request.args.get("months", 12, type=int)

    results = get_all_segments_roi(profiles, uplift, months)
    return jsonify(results)
