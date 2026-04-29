import joblib
import numpy as np

model = joblib.load("../models/churn_model.pkl")

def predict(data):
    return model.predict(np.array([data]))[0]