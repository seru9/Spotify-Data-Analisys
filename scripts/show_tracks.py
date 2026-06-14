import os
import pandas as pd

# do wykresu 1
# ==================================================================
# 1. PRZYGOTOWANIE LISTY ARTYSTÓW
# ==================================================================
target_artists = [
    "Brray",
    "Tony Dize",
    "Manuel Turizo",
    "Harry Styles",
    "Kendrick Lamar",
    "Rick Astley",
    "Taylor Swift",
]

os.makedirs("presentation", exist_ok=True)

# --- wczytanie danych ---
path = "data/clean_tracks.parquet"
if not os.path.exists(path):
    raise FileNotFoundError(
        f"Nie znaleziono pliku: {path} (sprawdź ścieżkę względną)"
    )

df = pd.read_parquet(path)

df = df.drop_duplicates(subset="track_id")

print("Wczytano df:", df.shape)
print("Dostępne kolumny w zbiorze:", list(df.columns))
print("-" * 70)

df_search = df.copy()

# Rozbijamy średniki i oczyszczamy z białych znaków (odpowiednik wyk. 06)
df_search["artist"] = df_search["artists"].str.split(";")
df_search = df_search.explode("artist")
df_search["artist"] = df_search["artist"].str.strip()

# ==================================================================
# 3. FILTROWANIE I WYŚWIETLANIE UTWORÓW
# ==================================================================
df_filtered = df_search[df_search["artist"].isin(target_artists)]

print(f"=== ZNALEZIONO UTWORY DLA WYBRANYCH ARTYSTÓW ===")
print(f"Łączna liczba wierszy w bazie dla tej grupy: {len(df_filtered)}\n")

# --- Dynamiczne wykrywanie nazwy kolumny z tytułem ---
title_column = None
for col in ["track_name", "name", "title"]:
    if col in df_filtered.columns:
        title_column = col
        break

if not title_column:
    raise KeyError(
        "Nie znaleziono kolumny z tytułem utworu (szukano: 'track_name', 'name', 'title')."
    )

# Definiujemy kolumny do wyświetlenia z poprawnie wykrytym tytułem i gatunkiem
columns_to_show = ["popularity", title_column, "artists", "track_genre"]

# Pętla wyświetlająca utwory dla każdego artysty po kolei
for artist in target_artists:
    # Wyciągamy utwory danego artysty i sortujemy od najwyższej popularności
    artist_tracks = df_filtered[df_filtered["artist"] == artist].sort_values(
        by="popularity", ascending=False
    )

    print("-" * 70)
    print(
        f"ARTYSTA: {artist.upper()} (Liczba utworów w bazie: {len(artist_tracks)})"
    )
    print("-" * 70)

    if artist_tracks.empty:
        print("  [Brak utworów w bazie danych - sprawdź pisownię!]")
    else:
        # Drukowanie sformatowanej tabeli dla artysty
        print(artist_tracks[columns_to_show].to_string(index=False))

    print("\n")