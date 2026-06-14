import pandas as pd
import numpy as np

df = pd.read_parquet("data/all_tracks.parquet")

# Diagnostyka przed czyszczeniem — wyk. 04: info() i describe()
# jako obowiązkowy krok wstępny przy każdej nowej ramce danych

print("=== df.info() ===")
df.info()

print("\n=== df.describe(include='all') ===")
print(df.describe(include="all"))

print(f"\nWymiar przed czyszczeniem: {df.shape}")

# Duplikaty — drop_duplicates()
# Ten sam utwór może pojawić się w wielu gatunkach (różne track_genre),
# więc usuwamy tylko wiersze identyczne we WSZYSTKICH kolumnach.

przed = len(df)
df = df.drop_duplicates()
print(f"Usunięto duplikatów: {przed - len(df)}")
df["key"] = df["key"].replace(-1, pd.NA)

# Usuwamy utwory krótsze niż 10 s i dłuższe niż 20 min 
przed = len(df)
df = df[(df["duration_ms"] > 10_000) & (df["duration_ms"] < 1_200_000)]
print(f"Usunięto outlierów duration_ms: {przed - len(df)}")

# Brakujące wartości
print("\n=== Brakujące wartości po czyszczeniu ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nWymiar po czyszczeniu: {df.shape}")

df.to_parquet("data/clean_tracks.parquet", index=False)
print("Zapisano: data/clean_tracks.parquet")

# Eksport CSV dla skryptu R (04_kmeans_clustering.R) — wyk. 07: to_csv()
df.to_csv("data/clean_tracks.csv", index=False)
print("Zapisano: data/clean_tracks.csv (dla skryptu R)")
