
# Your main.py code here...
from fastapi import FastAPI
import pickle
import numpy as np

# Load the trained Logistic Regression model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Earthquake Prediction API is running!"}

@app.get("/predict")
def predict(magnitude: float, depth: float, lat: float, lon: float):
    # Prepare input as a NumPy array
    test_input = np.array([[magnitude, depth, lat, lon]])
    prediction = model.predict(test_input)

    # Convert numerical prediction to label
    prediction_label = "Earthquake Expected" if prediction[0] == 1 else "No Earthquake"

    return {"prediction": prediction_label}
