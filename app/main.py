# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import StudentFeatures, predict_single

app = FastAPI(title="Student Score Predictor", version="0.1")

class PredictResponse(BaseModel):
    prediction: float

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Student Score Predictor API"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: StudentFeatures):
    try:
        pred = predict_single(payload)
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import requests

url = "http://0.0.0.0:8000/predict"
payload = {
  "hours_studied": 6,
  "attendance": 88,
  "assignments_completed": 7,
  "past_scores": 80
}
r = requests.post(url, json=payload)
print(r.json())
