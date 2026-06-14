import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("presentation", exist_ok=True)

# --- wczytanie danych ---
path = "data/clean_tracks.parquet"
if not os.path.exists(path):
    raise FileNotFoundError(f"Nie znaleziono pliku: {path} (sprawdź ścieżkę względną)")

df = pd.read_parquet(path)
print("Wczytano df:", df.shape)


# ==================================================================
# Rozbijamy kolumnę artists — str.split + str.strip
# ==================================================================
df_artists = df.copy().drop_duplicates(subset="track_id")
df_artists["artist"] = df_artists["artists"].str.split(";")
df_artists = df_artists.explode("artist")
df_artists["artist"] = df_artists["artist"].str.strip()
print("Po explode artystów:", df_artists.shape)

# ==================================================================
# DEFINICJA KRYTERIÓW I WYBRANYCH GWIAZD
# ==================================================================
MIN_TRACKS_OHW = 2
MAX_TRACKS_OHW = 10

famous_artists_list = [
    "Harry Styles",
    "Kendrick Lamar",
    "Rick Astley",
    "Taylor Swift",
]

# 1. Liczymy utwory dla każdego artysty za pomocą szybkiego .size()
artist_counts = df_artists.groupby("artist", observed=True).size()

# 2. Wybieramy artystów OHW ORAZ jawnie dopuszczamy nasze gwiazdy
valid_artists = artist_counts[
    ((artist_counts >= MIN_TRACKS_OHW) & (artist_counts <= MAX_TRACKS_OHW))
    | (artist_counts.index.isin(famous_artists_list))
].index

# 3. Filtrujemy główny DataFrame – teraz Taylor Swift i Kendrick też tu wejdą!
df_artists_filtered = df_artists[df_artists["artist"].isin(valid_artists)]
print(
    f"Liczba grup dla apply() zredukowana do {len(valid_artists)} (w tym wybrane gwiazdy)"
)


# FUNKCJA OBLICZAJĄCA GAP
def calculate_gap_metrics_optimized(group):
    pop = group["popularity"].to_numpy()

    max_pop = pop.max()

    # Mediana z pozostałych utworów (wycinamy jedno najwyższe wystąpienie)
    mask = np.ones(len(pop), dtype=bool)
    mask[np.argmax(pop)] = False
    rest_pop = pop[mask]

    # eśli artysta miałby tylko 1 utwór (Rick Astley :D)
    if len(rest_pop) == 0:
        median_rest = max_pop
    else:
        median_rest = np.median(rest_pop)

    gap = max_pop - median_rest

    return pd.Series({"gap": gap, "track_count": len(pop)})


# Obliczamy metryki dla przefiltrowanej grupy
artist_metrics = (
    df_artists_filtered.groupby("artist", observed=True)
    .apply(calculate_gap_metrics_optimized, include_groups=False)
    .reset_index()
)

# ==================================================================
# PRZYGOTOWANIE DANYCH DO WYKRESU
# ==================================================================

# 1. Sektujemy tylko "prawdziwych" kandydatów na OHW do Top 3 (czyli tych z track_count <= 10)
ohw_candidates = artist_metrics[artist_metrics["track_count"] <= MAX_TRACKS_OHW]
top_3_ohw = ohw_candidates.sort_values("gap", ascending=False).head(3)

# 2. Wyciągamy dane dla naszych gwiazd (one mogą mieć track_count > 10)
famous_artists_data = artist_metrics[
    artist_metrics["artist"].isin(famous_artists_list)
]

# 3. Łączymy w jeden DataFrame pod wykres
ohw_ranking = pd.concat([top_3_ohw, famous_artists_data]).drop_duplicates(
    subset=["artist"]
)

# Podgląd w konsoli
print("\n--- Zestawienie po poprawce: Top 3 OHW vs Globalne Gwiazdy ---")
print(ohw_ranking[["artist", "gap", "track_count"]].to_string(index=False))

# ==================================================================
# RYSOWANIE WYKRESU
# ==================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Dynamiczne kolorowanie: złoty dla Top 3 OHW, niebieski dla reszty
top_3_names = top_3_ohw["artist"].tolist()
colors = [
    "gold" if artist in top_3_names else "cornflowerblue"
    for artist in ohw_ranking["artist"]
]

ax.bar(ohw_ranking["artist"], ohw_ranking["gap"], color=colors)

ax.set_title(
    "Kontrast: Top 3 One-Hit Wonders vs Globalne Gwiazdy (Poprawione)",
    fontsize=14,
    pad=15,
)
ax.set_ylabel("GAP = max(pop) - mediana(reszta)", fontsize=12)

# Dynamiczne dopasowanie osi Y (uwzględnia wartości ujemne)
min_gap = min(0, ohw_ranking["gap"].min() * 1.1)
max_gap = ohw_ranking["gap"].max() * 1.1
ax.set_ylim(min_gap, max_gap)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.tight_layout()
plt.savefig("presentation/wykres_1.png", dpi=150)
plt.close()

print("\nZapisano poprawny wykres w: presentation/wykres_1.png")

# WYKRES 2: Box plot — rozkład popularity per genre (top 14 gatunków + pop)

top_genres = df["track_genre"].value_counts().head(14).index.tolist()
if "pop" not in top_genres:
    top_genres.append("pop")
df_top = df[df["track_genre"].isin(top_genres)]

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df_top, x="track_genre", y="popularity", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_xlabel("Gatunek muzyczny")
ax.set_ylabel("Popularność")
ax.set_title("Rozkład popularności w top 14 gatunkach + pop")
plt.tight_layout()
plt.savefig("presentation/wykres_2.png", dpi=150)
plt.close()
print("Zapisano: wykres_2.png")