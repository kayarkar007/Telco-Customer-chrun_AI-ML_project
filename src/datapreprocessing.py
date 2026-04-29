import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    path = os.path.join(BASE_DIR, "data/raw_data/Telco-Customer-Churn.csv")
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Fix datatype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing
    df = df.dropna()

    # Drop useless column
    df = df.drop("customerID", axis=1)

    return df

def save_clean_data(df):
    path = os.path.join(BASE_DIR, "data/clean_data/clean.csv")
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    save_clean_data(df)

    print("Cleaning Done:", df.shape)