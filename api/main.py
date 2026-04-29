from fastapi import FastAPI
import joblib
import numpy as np
import os
import json
import pandas as pd

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
model = joblib.load(model_path)

# Load feature columns
columns_path = os.path.join(BASE_DIR, "models", "columns.json")

with open(columns_path, "r") as f:
    model_columns = json.load(f)


@app.get("/")
def home():
    return {"message": "Churn Model Running"}


@app.get("/predict")
def predict(
    tenure: int,
    MonthlyCharges: float,
    TotalCharges: float,
    SeniorCitizen: int = 0
):
    # Create base input
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "SeniorCitizen": SeniorCitizen
    }

    # Convert to dataframe
    df = pd.DataFrame([input_data])

    # Align columns with training
    df = pd.get_dummies(df)

    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]

    return {"churn": int(prediction)}