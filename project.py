from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# FastAPI Object
app = FastAPI()

# Load Trained Model
model = load("salary_model.joblib")

# Input Schema
class Employee(BaseModel):
    experience: float

# Home Route
@app.get("/")
def home():
    return {
        "message": "Salary Prediction API Running"
    }

# Prediction Route
@app.post("/predict")
def predict_salary(data: Employee):

    exp = np.array([[data.experience]])

    prediction = model.predict(exp)

    return {
        "experience": data.experience,
        "predicted_salary": float(prediction[0])
    }