import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, filename: str = "model.joblib"):
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    return filepath

def load_model(filename: str = "model.joblib"):
    filepath = MODELS_DIR / filename
    return joblib.load(filepath)
