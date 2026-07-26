<div align="center">

# Customer Lifetime Value & Segmentation Engine

**A production-grade CLV prediction system combining unsupervised segmentation, probabilistic cold-start modeling, SHAP explainability, and AI-generated retention strategy — served through a Flask dashboard and REST API.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat-square)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=flat-square)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[Live Demo](#) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>

---

## Overview

Most CLV projects stop at a K-Means plot and a linear regression line. This one is built the way a production analytics team would build it: RFM feature engineering, model comparison and selection, a real solution for customers with almost no purchase history, model interpretability that a business stakeholder can actually read, and an AI layer that turns predictions into action.

It runs as a self-contained Flask application — trainable pipeline, REST API, and dashboard, all in one repo.

## Why This Is Different

| | Typical CLV project | This project |
|---|---|---|
| **Segmentation** | K-Means on 2–3 RFM fields | K-Means on 6 engineered features with persona assignment |
| **CLV prediction** | Single linear regression | XGBoost vs. Random Forest vs. Ridge, best model auto-selected |
| **New customers** | Ignored or set to zero | Hybrid BG/NBD + Gamma-Gamma model blended with ML output |
| **Explainability** | A feature-importance bar chart | SHAP at the global, segment, and individual-customer level |
| **Business layer** | None | GenAI retention copy + interactive ROI calculator |
| **Architecture** | Single Jupyter notebook | MVC Flask app with versioned REST endpoints |

## Results

| Metric | Value |
|---|---|
| Transactions processed | 60,000 |
| Unique customers | 31,154 |
| Customers with a single order (cold-start) | 13,322 (42.8%) |
| Segmentation quality (Silhouette Score) | *populated after training* |
| Best CLV model | *auto-selected — XGBoost / RF / Ridge* |
| R² | *populated after training* |
| RMSE | *populated after training* |
| MAPE | *populated after training* |

Run `python train_pipeline.py` to populate these from your own data.

## How It Works

```
CSV data
   │
   ▼
Data pipeline  →  clean, validate, deduplicate
   │
   ▼
Feature engineering  →  RFM + tenure, return rate, category diversity
   │
   ├──▶ K-Means segmentation  →  business persona labels
   │
   ├──▶ CLV regression  →  XGBoost / RF / Ridge, best model kept
   │
   └──▶ Cold-start model  →  BG/NBD + Gamma-Gamma, blended by order count
                │
                ▼
        SHAP explainability
                │
                ▼
        GenAI retention advisor (Groq)
                │
                ▼
        Flask dashboard + REST API
```

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-Learn, XGBoost |
| Probabilistic modeling | `lifetimes` (BG/NBD, Gamma-Gamma) |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn, Chart.js |
| Web layer | Flask, Jinja2 |
| GenAI | Groq API (Llama 3.3 70B) |
| Deployment | Render |

## Project Structure

```
├── app.py                      # Flask entry point
├── config.py                   # Centralized configuration
├── train_pipeline.py           # End-to-end training orchestrator
├── requirements.txt
├── Procfile
│
├── data/
│   ├── shopify_sales_dataset_ml_eda.csv
│   └── processed/               # Generated features & predictions
│
├── src/
│   ├── data_pipeline.py         # Load, validate, clean CSV
│   ├── feature_engineering.py   # RFM + behavioral features
│   ├── clustering.py            # K-Means + segment profiling
│   ├── clv_model.py             # XGBoost / RF / Ridge comparison
│   ├── cold_start.py            # BG/NBD + Gamma-Gamma hybrid
│   ├── explainability.py        # SHAP analysis
│   ├── genai_advisor.py         # LLM retention strategies
│   └── roi_calculator.py        # Revenue impact estimation
│
├── routes/
│   ├── dashboard.py             # Page routes
│   ├── api.py                   # REST API endpoints
│   └── advisor.py               # GenAI API endpoint
│
├── models/                      # Trained model artifacts (.pkl)
├── templates/                   # Jinja2 HTML templates
├── static/                      # CSS, JS, images
└── notebooks/                   # Exploratory analysis
```

## Getting Started

### Prerequisites
- Python 3.12+
- (Optional) A [Groq API key](https://console.groq.com/) for the GenAI retention advisor

### Installation

```bash
git clone https://github.com/Adityya21/ecommerce-clv-segmentation.git
cd ecommerce-clv-segmentation
pip install -r requirements.txt
```

### Configuration (optional — GenAI features)

```bash
cp .env.example .env
# add your GROQ_API_KEY to .env
```

### Train the models

```bash
python train_pipeline.py
```

This runs the full pipeline end to end: cleaning, RFM engineering, segmentation, CLV regression, cold-start model training, and SHAP analysis. Artifacts are written to `models/` and `data/processed/`.

### Launch the dashboard

```bash
python app.py
```

Visit `http://localhost:5000`.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | `POST` | Predict CLV for a given customer |
| `/api/segments` | `GET` | Retrieve all segment profiles |
| `/api/dashboard-data` | `GET` | Chart data for the dashboard |
| `/api/roi` | `POST` | Calculate retention ROI for a segment |
| `/api/roi/all` | `GET` | ROI estimate across all segments |
| `/api/advisor` | `POST` | Generate an AI-written retention strategy |

## Methodology

**RFM feature engineering** — 60K transactions are aggregated into 31K customer profiles across 15+ features: Recency, Frequency, Monetary value (net of returns), Tenure, Return Rate, Category Diversity, Average Discount, Average Rating, and Purchase Rate.

**Segmentation** — K-Means on 6 scaled features (Recency, Frequency, Monetary, Tenure, Return Rate, Category Diversity). Optimal *k* is chosen via the Elbow method and Silhouette Score, and each resulting cluster is mapped to a business persona.

**CLV regression** — XGBoost, Random Forest, and Ridge are trained and compared; the best performer is auto-selected by R² and reported with a plain-language metric interpretation.

**Cold-start CLV** — the hardest part of any CLV system. For the 13K+ customers with only one or two orders, standard ML has nothing to learn from. This pipeline blends a probabilistic BG/NBD model (purchase frequency) and a Gamma-Gamma model (transaction value) with the ML prediction, weighting the blend by how much transaction history each customer actually has.

**Explainability** — a SHAP TreeExplainer sits on top of the CLV model, producing global feature importance, per-segment drivers, and individual per-customer waterfall explanations.

**GenAI retention advisor** — a Groq-hosted LLM turns each segment's profile and CLV prediction into concrete retention tactics and marketing copy.

## Reading the Metrics

| Metric | What it tells a stakeholder |
|---|---|
| RMSE | Typical dollar error per customer prediction |
| R² | Share of CLV variation the model explains |
| MAPE | Typical percentage error, useful as a rough confidence band |
| Silhouette Score | How distinct the customer segments are — higher means cleaner separation |

## License

Distributed under the [MIT License](LICENSE).

## Author

**Aditya**
B.Tech Computer Engineering — AI/ML & Data Science

[GitHub](https://github.com/Adityya21) · [LinkedIn](https://linkedin.com/in/yourprofile)
