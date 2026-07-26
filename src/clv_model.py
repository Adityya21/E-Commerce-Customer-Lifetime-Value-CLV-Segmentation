"""
CLV Regression Model Module
=============================
Trains and evaluates XGBoost, Random Forest, and Ridge regression
models to predict Customer Lifetime Value from engineered features.

Includes business-friendly metric interpretation and model persistence.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Features for CLV prediction (excluding target and IDs)
NUMERIC_FEATURES = [
    "recency", "frequency", "monetary", "tenure_days",
    "avg_discount", "avg_quantity", "avg_rating", "avg_shipping",
    "return_rate", "total_items", "n_returns", "category_diversity",
    "avg_product_price", "purchase_rate", "avg_order_value",
]

CATEGORICAL_FEATURES = [
    "dominant_category", "dominant_channel", "country", "dominant_payment",
]

TARGET = "total_clv"


def prepare_features(rfm_df):
    """
    Prepare feature matrix and target for CLV prediction.
    Handles label encoding for categoricals.

    Returns
    -------
    X, y, feature_names, label_encoders
    """
    df = rfm_df.copy()

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    encoded_cat_cols = [c + "_encoded" for c in CATEGORICAL_FEATURES]
    feature_cols = NUMERIC_FEATURES + encoded_cat_cols

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Handle any infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y, feature_cols, label_encoders


def train_and_compare(X, y, test_size=0.2, random_state=42):
    """
    Train XGBoost, Random Forest, and Ridge — compare and pick best.

    Returns
    -------
    dict with best model, scaler, metrics, all_results
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Ridge": Ridge(alpha=1.0),
    }

    results = {}
    for name, model in models.items():
        print(f"\n[CLV Model] Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Clip negative predictions to 0
        y_pred = np.clip(y_pred, 0, None)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        # MAPE: avoid division by zero for customers with 0 CLV
        mask = y_test > 0
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100

        results[name] = {
            "model": model,
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
            "y_test": y_test,
            "y_pred": y_pred,
        }

        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.1f}%")

    # Pick best model by R²
    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\n[CLV Model] ✅ Best model: {best_name} (R²={results[best_name]['r2']:.4f})")

    return {
        "best_name": best_name,
        "best_model": results[best_name]["model"],
        "scaler": scaler,
        "metrics": {name: {k: v for k, v in r.items() if k not in ("model", "y_test", "y_pred")}
                    for name, r in results.items()},
        "best_metrics": results[best_name],
        "all_results": results,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
    }


def interpret_metrics(metrics):
    """
    Generate business-friendly interpretations of model metrics.

    Returns
    -------
    dict: metric_name → business interpretation string
    """
    best = None
    best_name = None
    for name, m in metrics.items():
        if best is None or m["r2"] > best["r2"]:
            best = m
            best_name = name

    interpretations = {
        "rmse": (
            f"On average, our CLV prediction is off by ${best['rmse']:,.0f} per customer. "
            f"For a customer with a predicted CLV of $2,000, the actual value is likely "
            f"between ${2000 - best['rmse']:,.0f} and ${2000 + best['rmse']:,.0f}."
        ),
        "r2": (
            f"Our {best_name} model explains {best['r2']*100:.1f}% of the variation in "
            f"customer lifetime value. {'This is strong predictive power.' if best['r2'] > 0.7 else 'There is room for improvement with additional features.'}"
        ),
        "mape": (
            f"The typical prediction error is {best['mape']:.1f}%. For example, a customer "
            f"predicted to have a $1,000 CLV might actually have a CLV between "
            f"${1000*(1-best['mape']/100):,.0f} and ${1000*(1+best['mape']/100):,.0f}."
        ),
    }
    return interpretations


def save_model_artifacts(result, label_encoders, feature_cols):
    """Save the best model, scaler, and encoders."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["best_model"], config.CLV_MODEL_PATH)
    joblib.dump(result["scaler"], config.SCALER_PATH)
    joblib.dump(label_encoders, config.LABEL_ENCODER_PATH)
    joblib.dump(feature_cols, config.FEATURE_COLUMNS_PATH)

    print(f"[CLV Model] Saved: {result['best_name']} model, scaler, encoders")


def predict_clv(features_dict, model=None, scaler=None, label_encoders=None, feature_cols=None):
    """
    Predict CLV for a single customer given their features.

    Parameters
    ----------
    features_dict : dict
        Customer feature values.

    Returns
    -------
    float: predicted CLV
    """
    if model is None:
        model = joblib.load(config.CLV_MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(config.SCALER_PATH)
    if label_encoders is None:
        label_encoders = joblib.load(config.LABEL_ENCODER_PATH)
    if feature_cols is None:
        feature_cols = joblib.load(config.FEATURE_COLUMNS_PATH)

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in features_dict:
            le = label_encoders[col]
            try:
                features_dict[col + "_encoded"] = le.transform([str(features_dict[col])])[0]
            except ValueError:
                # Unknown category — use most frequent
                features_dict[col + "_encoded"] = 0

    X = pd.DataFrame([features_dict])[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    return max(0, round(float(pred), 2))
