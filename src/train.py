import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from feature_engineering import process_features

df = pd.read_csv("../data/raw_data/clean.csv")

X, y = process_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

joblib.dump(model, "../models/churn_model.pkl")

print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))