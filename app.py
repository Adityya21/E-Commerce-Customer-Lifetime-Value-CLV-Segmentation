"""
Flask Application Entry Point
================================
CLV Intelligence — E-Commerce Customer Lifetime Value
& Segmentation Dashboard

Usage:
    python app.py          # Development server
    gunicorn app:app       # Production (Render/Railway)
"""

import os
import sys
from pathlib import Path
from flask import Flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config


def create_app():
    """Flask application factory."""
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )

    app.secret_key = config.SECRET_KEY

    # Register blueprints
    from routes.dashboard import dashboard_bp
    from routes.api import api_bp
    from routes.advisor import advisor_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(advisor_bp)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=config.DEBUG,
    )
