import os
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score


# --------------------------------------------------------
# DATASET CANDIDATE PATHS
# --------------------------------------------------------
CANDIDATE_DATA_PATHS = [
    "./data/himalaya.csv",
    "./data/himalayan.csv",
    "./data/himalaya.csv".replace("/", os.sep),
    "./data/himalayan.csv".replace("/", os.sep),
    "./himalaya.csv",
    "./himalayan.csv",
    "/mnt/data/Himalayan_Avalanche_Dataset_v1 (1).csv",
    "/mnt/data/himalaya.csv",
]

# Allowed locations
LOCATIONS = [
    "Leh (Ladakh)",
    "Gulmarg (J&K)",
    "Manali (Himachal Pradesh)",
    "Shimla (Himachal Pradesh)",
    "Rohtang Pass (Himachal Pradesh)",
    "Auli (Uttarakhand)",
    "Kedarnath (Uttarakhand)",
    "Gangotri (Uttarakhand)",
    "Nanda Devi (Uttarakhand)",
    "Solang Valley (Himachal Pradesh)"
]

MODEL_FILE = "rf_pipeline.joblib"


# --------------------------------------------------------
# FLASK APP SETUP
# --------------------------------------------------------
app = Flask(__name__, static_folder="templates")
CORS(app)


# --------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------
def find_dataset_path(candidates):
    """Return the first existing path from candidates."""
    for p in candidates:
        p_expanded = os.path.expanduser(os.path.expandvars(p))
        if os.path.exists(p_expanded):
            return os.path.abspath(p_expanded)
    return None


DATA_PATH = find_dataset_path(CANDIDATE_DATA_PATHS)
if DATA_PATH:
    print(f"[startup] Using dataset at: {DATA_PATH}")
else:
    print("[startup] No dataset found.")


def load_and_prepare_dataset(path):
    """Load the dataset and prepare the target column."""
    if path is None:
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["aspect_mean", "elevation_mean", "elevation_sum", "slope_mean", "slope_sum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "location_name" not in df.columns:
        df["location_name"] = np.nan

    if "avalanche_occurred" in df.columns:
        df["avalanche_occurred"] = df["avalanche_occurred"].astype(int).fillna(0)
    else:
        warnings.warn("No target found. Creating heuristic label.")
        df["avalanche_occurred"] = (
            ((df["slope_mean"] >= 30) & (df["slope_mean"] <= 45) & (df["elevation_mean"] >= 3000))
            | (df["slope_mean"] >= 40)
        ).astype(int)

    df = df[required + ["location_name", "avalanche_occurred"]]
    df["location_name"] = df["location_name"].fillna("Unknown").astype(str)

    return df


def make_onehot_encoder_compatible(**kwargs):
    """Fix for sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_and_train_pipeline(df):
    """Build preprocessing + Random Forest model."""
    X = df.drop(columns=["avalanche_occurred"])
    y = df["avalanche_occurred"].astype(int)

    numeric = ["aspect_mean", "elevation_mean", "elevation_sum", "slope_mean", "slope_sum"]
    cat = ["location_name"]

    numeric_transform = Pipeline([("scaler", StandardScaler())])
    categorical_transform = Pipeline([("onehot", make_onehot_encoder_compatible())])

    preprocessor = ColumnTransformer([
        ("num", numeric_transform, numeric),
        ("cat", categorical_transform, cat)
    ], sparse_threshold=0)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    stratify = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    pipeline.fit(X_train, y_train)

    try:
        y_pred = pipeline.predict(X_test)
        print("[model] Accuracy:", accuracy_score(y_test, y_pred))
    except:
        pass

    return pipeline


# --------------------------------------------------------
# MODEL LOADING / TRAINING
# --------------------------------------------------------
pipeline = None

if os.path.exists(MODEL_FILE):
    try:
        pipeline = joblib.load(MODEL_FILE)
        print("[model] Loaded existing model.")
    except Exception:
        pipeline = None

if pipeline is None:
    df = load_and_prepare_dataset(DATA_PATH)
    pipeline = build_and_train_pipeline(df)
    joblib.dump(pipeline, MODEL_FILE)
    print("[model] Saved trained model.")


# --------------------------------------------------------
# RISK STAGE FUNCTION
# --------------------------------------------------------
def risk_stage_from_prob(p):
    if p < 0.25: return "Low"
    if p < 0.5: return "Moderate"
    if p < 0.75: return "High"
    return "Very High"


# --------------------------------------------------------
# PREDICT API (UPDATED WITH CONFIDENCE SCORE)
# --------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model is not available"}), 500

    data = request.get_json()

    required = ["aspect_mean", "elevation_mean", "elevation_sum", "slope_mean", "slope_sum", "location_name"]
    missing = [k for k in required if k not in data or data[k] == ""]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Convert inputs
    try:
        sample = {
            "aspect_mean": float(data["aspect_mean"]),
            "elevation_mean": float(data["elevation_mean"]),
            "elevation_sum": float(data["elevation_sum"]),
            "slope_mean": float(data["slope_mean"]),
            "slope_sum": float(data["slope_sum"]),
            "location_name": str(data["location_name"]).strip()
        }
    except Exception as e:
        return jsonify({"error": f"Invalid numeric input: {e}"}), 400

    if sample["location_name"] not in LOCATIONS:
        return jsonify({"error": f"location_name must be one of: {LOCATIONS}"}), 400

    X_sample = pd.DataFrame([sample])

    try:
        proba = pipeline.predict_proba(X_sample)[0][1]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    pred_class = int(proba >= 0.5)
    pct = round(proba * 100, 2)
    stage = risk_stage_from_prob(proba)

    # ⭐ NEW: Confidence Score
    confidence = round(max(proba, 1 - proba) * 100, 2)

    return jsonify({
        "predicted_class": pred_class,
        "probability": float(proba),
        "percentage": pct,
        "risk_stage": stage,
        "confidence": confidence   # ⭐ Added field
    })


# --------------------------------------------------------
# FRONTEND FILE SERVING
# --------------------------------------------------------
@app.route("/", defaults={"path": "prediction.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    safe_path = path if path else "prediction.html"
    templates_dir = os.path.join(os.getcwd(), "templates")
    full = os.path.join(templates_dir, safe_path)

    if os.path.exists(full):
        return send_from_directory(templates_dir, safe_path)

    return "File not found", 404


# --------------------------------------------------------
# RUN APP
# --------------------------------------------------------
if __name__ == "__main__":
    print("[startup] Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
