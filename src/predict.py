# src/predict.py
import sys
from pathlib import Path
import pandas as pd
from pydantic import BaseModel

# ensure src package import works when running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from model_utils import load_model

# Load model once at import time (so API endpoints are fast)
MODEL = load_model(filename="student_score_model.joblib")

# Define input schema expected by the model
class StudentFeatures(BaseModel):
    hours_studied: float
    attendance: float
    assignments_completed: float
    past_scores: float

def predict_single(features: StudentFeatures):
    """Return model prediction (single example)."""
    # create DataFrame with same column order used during training
    X = pd.DataFrame([{
        "hours_studied": features.hours_studied,
        "attendance": features.attendance,
        "assignments_completed": features.assignments_completed,
        "past_scores": features.past_scores
    }])
    preds = MODEL.predict(X)
    # return scalar
    return float(preds[0])
