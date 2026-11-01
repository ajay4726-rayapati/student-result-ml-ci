
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Model file location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_result_model.pkl"

# Input features expected by the ML model
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


def load_model():
    """Load the trained student result prediction model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Ensure student_result_model.pkl "
            "is available or run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# Health endpoint for Docker and CI/CD testing
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


# Root endpoint to check whether the application is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "student-result-prediction",
        "status": "ok",
    }), 200


# Student result prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON request body is required"
        }), 400

    # Validate required input features
    missing = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing,
        }), 400

    # Convert feature values to numbers
    try:
        values = {
            feature: float(data[feature])
            for feature in FEATURES
        }

        if not all(
            pd.notna(value) and value not in (float("inf"), float("-inf"))
            for value in values.values()
        ):
            raise ValueError("Values must be finite numbers")

    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "error": "All feature values must be valid finite numbers"
        }), 400

    # Load model and generate prediction
    try:
        model = load_model()
        sample = pd.DataFrame([values], columns=FEATURES)
        code = int(model.predict(sample)[0])

    except FileNotFoundError as exc:
        app.logger.error("Model loading failed: %s", exc)
        return jsonify({
            "error": "Trained model file is unavailable"
        }), 500

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Unable to generate prediction"
        }), 500

    return jsonify({
        "prediction": "PASS" if code == 1 else "FAIL",
        "prediction_code": code,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
