import pandas as pd
import glob
import os

# glob.glob() — wyk. 07: automatyzacja przetwarzania plików w katalogu
# Szukamy wszystkich plików .csv w katalogu data/
csv_files = glob.glob(os.path.join("data", "*.csv"))

print(f"Znaleziono {len(csv_files)} plików CSV.")

dfs = []
for path in csv_files:
    # Pobieramy samą nazwę pliku z pełnej ścieżki
    file_name = os.path.basename(path)
    
    if file_name.endswith("Liked_Songs.csv"):
        print(f"  Pominięto: {file_name} (plik ignorowany)")
        continue  
        
    df = pd.read_csv(path)   # wyk. 07: pd.read_csv w pętli po plikach
    dfs.append(df)
    print(f"  Wczytano: {file_name} — {len(df)} wierszy")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nŁącznie wierszy po scaleniu: {len(df_all)}")
print(f"Kolumny: {df_all.columns.tolist()}")

# Sprawdzamy katalog wyjściowy przed zapisem — wyk. 07: os.path.exists + os.mkdir
if not os.path.exists("data"):
    os.mkdir("data")

df_all.to_parquet("data/all_tracks.parquet", index=False)
print("\nZapisano: data/all_tracks.parquet")
