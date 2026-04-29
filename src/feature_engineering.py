import pandas as pd

def process_features(df):
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    return X, y