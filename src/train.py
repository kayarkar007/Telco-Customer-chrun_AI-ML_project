import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import process_features

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct data path
data_path = os.path.join(BASE_DIR, "data", "clean_data", "clean.csv")

df = pd.read_csv(data_path)

# Features
X, y = process_features(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import json

# Save feature columns
columns_path = os.path.join(BASE_DIR, "models", "columns.json")

with open(columns_path, "w") as f:
    json.dump(list(X.columns), f)
    
# Train
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
joblib.dump(model, model_path)

print("Train Score:", model.score(X_train, y_train)*100)
print("Test Score:", model.score(X_test, y_test)*100)
print("Model saved at:", model_path)