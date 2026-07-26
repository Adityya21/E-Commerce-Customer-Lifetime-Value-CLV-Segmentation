"""
Dashboard Routes
=================
Serves the main dashboard and segment pages with pre-loaded data.
"""

import json
from flask import Blueprint, render_template
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.roi_calculator import get_all_segments_roi

dashboard_bp = Blueprint("dashboard", __name__)


def _load_json(path):
    """Safely load a JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@dashboard_bp.route("/")
def index():
    """Landing page / main dashboard."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    metrics = _load_json(config.PROCESSED_DIR / "model_metrics.json")
    shap_data = _load_json(config.PROCESSED_DIR / "shap_analysis.json")

    # Calculate total revenue and customer count
    total_customers = sum(p["size"] for p in profiles.values())
    total_revenue = sum(p["total_revenue"] for p in profiles.values())
    avg_clv = total_revenue / max(total_customers, 1)

    # ROI summary at 15% retention uplift
    roi_data = get_all_segments_roi(profiles, retention_uplift_pct=15, months=12)
    total_revenue_at_risk = sum(r["revenue_at_risk"] for r in roi_data)

    return render_template(
        "dashboard.html",
        profiles=profiles,
        metrics=metrics,
        shap_data=shap_data,
        total_customers=total_customers,
        total_revenue=total_revenue,
        avg_clv=avg_clv,
        total_revenue_at_risk=total_revenue_at_risk,
        roi_data=roi_data,
    )


@dashboard_bp.route("/segments")
def segments():
    """Detailed segment profiles page."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    shap_data = _load_json(config.PROCESSED_DIR / "shap_analysis.json")

    return render_template(
        "segments.html",
        profiles=profiles,
        shap_data=shap_data,
    )


@dashboard_bp.route("/predict")
def predict_page():
    """CLV prediction form page."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    return render_template("predict.html", profiles=profiles)


@dashboard_bp.route("/advisor")
def advisor_page():
    """GenAI retention advisor page."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    shap_data = _load_json(config.PROCESSED_DIR / "shap_analysis.json")
    return render_template("advisor.html", profiles=profiles, shap_data=shap_data)


@dashboard_bp.route("/roi")
def roi_page():
    """ROI calculator page."""
    profiles = _load_json(config.SEGMENT_PROFILES_PATH)
    return render_template("roi.html", profiles=profiles)
