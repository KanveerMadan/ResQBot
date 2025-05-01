from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

class EarthquakeData(BaseModel):
    latitude: float
    longitude: float
    depth: float
    mag: float
    place_length: int
    status_encoded: int
    location_source_encoded: int

# Check model file existence
model_path = "logistic_regression24_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")

@app.post("/predict")
def predict(data: EarthquakeData):
    try:
        features = np.array([[
            data.latitude,
            data.longitude,
            data.depth,
            data.mag,
            data.place_length,
            data.status_encoded,
            data.location_source_encoded
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return {
            "earthquake_predicted": bool(prediction),
            "confidence": round(probability, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
