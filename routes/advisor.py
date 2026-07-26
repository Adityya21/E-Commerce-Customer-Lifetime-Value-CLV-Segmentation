"""
GenAI Advisor Routes
=====================
Endpoint for generating LLM-powered retention strategies.
"""

import json
from flask import Blueprint, request, jsonify
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.genai_advisor import generate_retention_strategy

advisor_bp = Blueprint("advisor", __name__, url_prefix="/api")


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@advisor_bp.route("/advisor", methods=["POST"])
def advisor():
    """
    Generate AI-powered retention strategy for a segment.

    Expects JSON:
    {
        "segment_id": 0,
        "provider": "groq"  // optional, defaults to config
    }

    Returns LLM-generated strategy JSON.
    """
    try:
        data = request.get_json()
        segment_id = str(data.get("segment_id", 0))
        provider = data.get("provider", None)

        profiles = _load_json(config.SEGMENT_PROFILES_PATH)
        if segment_id not in profiles:
            return jsonify({"error": f"Unknown segment: {segment_id}"}), 400

        profile = profiles[segment_id]

        # Load SHAP drivers for this segment
        shap_data = _load_json(config.PROCESSED_DIR / "shap_analysis.json")
        shap_drivers = None
        if "segment_shap" in shap_data and segment_id in shap_data["segment_shap"]:
            shap_drivers = shap_data["segment_shap"][segment_id].get("top_drivers", [])

        result = generate_retention_strategy(profile, shap_drivers, provider)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
