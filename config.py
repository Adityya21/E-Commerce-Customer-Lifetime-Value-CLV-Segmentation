import os
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "shopify_sales_dataset_ml_eda.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# ─── Model Files ────────────────────────────────────────────────────
KMEANS_PATH = MODELS_DIR / "kmeans_model.pkl"
CLV_MODEL_PATH = MODELS_DIR / "clv_xgboost.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoders.pkl"
BGNBD_PATH = MODELS_DIR / "bgnbd_model.pkl"
GAMMA_GAMMA_PATH = MODELS_DIR / "gamma_gamma_model.pkl"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"
SEGMENT_PROFILES_PATH = PROCESSED_DIR / "segment_profiles.json"
CLUSTER_SCALER_PATH = MODELS_DIR / "cluster_scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# ─── Processed Data ─────────────────────────────────────────────────
RFM_CSV = PROCESSED_DIR / "rfm_features.csv"
SEGMENTS_CSV = PROCESSED_DIR / "customer_segments.csv"
CLV_PREDICTIONS_CSV = PROCESSED_DIR / "clv_predictions.csv"
SHAP_VALUES_PATH = PROCESSED_DIR / "shap_values.pkl"

# ─── GenAI Config ───────────────────────────────────────────────────
GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "groq")  # "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── App Config ─────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "clv-segmentation-dev-key-2026")
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

# ─── Segment Personas ──────────────────────────────────────────────
SEGMENT_PERSONAS = {
    0: {"name": "Premium Loyalists", "emoji": "🏆", "color": "#ffd700"},
    1: {"name": "At-Risk Whales", "emoji": "⚠️", "color": "#ff6b6b"},
    2: {"name": "Dormant Browsers", "emoji": "💤", "color": "#78909c"},
    3: {"name": "Rising Champions", "emoji": "🌱", "color": "#66bb6a"},
    4: {"name": "Deal Seekers", "emoji": "🏷️", "color": "#4fc3f7"},
}
