import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/clean.csv")

print(df["Churn"].value_counts(normalize=True))

df.groupby("Churn")["tenure"].mean().plot(kind="bar")
plt.show()

df.groupby("Churn")["MonthlyCharges"].mean().plot(kind="bar")
plt.show()

df.groupby("Churn")["TotalCharges"].mean().plot(kind="bar")
plt.show()