
from fastapi import FastAPI
import pickle
import numpy as np
import os
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Try loading the model with error handling
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Error: Model file not found. Make sure 'logistic_regression_model.pkl' is available.")

# Define input schema for prediction
class EarthquakeInput(BaseModel):
    magnitude: float
    depth: float
    lat: float
    lon: float

@app.post("/predict")
def predict(data: EarthquakeInput):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}
    
    test_input = np.array([[data.magnitude, data.depth, data.lat, data.lon]])
    prediction = model.predict(test_input)
    prediction_label = "Earthquake Expected" if prediction[0] == 1 else "No Earthquake"
    
    return {"prediction": prediction_label}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Railway-provided PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
