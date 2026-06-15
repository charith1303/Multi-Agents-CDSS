import pandas as pd

df = pd.read_csv("data/dataset.csv")

disease = "Urinary tract infection"

rows = df[df["Disease"] == disease]

print(rows.head())