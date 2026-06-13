import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("presentation", exist_ok=True)

df = pd.read_parquet("data/clean_tracks.parquet")

audio_features = ["danceability", "energy", "acousticness",
                  "valence", "instrumentalness", "speechiness"]

# ==================================================================
# WYKRES 3: Top 20 gatunków wg średniej valence ("najweselsze")
# ==================================================================

# groupby + reset_index + sort_values — wyk. 05
# Analogia: wynik = tips.groupby('day')['total_bill'].mean()
#           wynik = wynik.reset_index()
#           wynik = wynik.sort_values(...)
valence_by_genre = df.groupby("track_genre", observed=True)["valence"].mean()
valence_by_genre = valence_by_genre.reset_index()
valence_by_genre = valence_by_genre.sort_values("valence", ascending=False)
top20_valence = valence_by_genre.iloc[:20, :]  # wyk. 04: iloc po pozycji

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_valence["track_genre"], top20_valence["valence"], color="gold")
ax.set_xlabel("Średnia valence (pozytywny wydźwięk)")
ax.set_title("Top 20 najweselszych gatunków muzycznych")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("presentation/wykres4_top20_valence.png", dpi=150)
plt.close()
print("Zapisano: wykres4_top20_valence.png")

# ==================================================================
# WYKRES 4: Udział explicit per gatunek (Top 20 pod względem wulgaryzmów)
# ==================================================================

# 1. Zliczamy kombinacje gatunek + explicit
explicit_counts = df.groupby("track_genre", observed=True)["explicit"].value_counts().reset_index(name="count")

# 2. Tworzymy tabelę przestawną (pivot) dla WSZYSTKICH dostępnych gatunków
explicit_pivot = explicit_counts.pivot(index="track_genre", columns="explicit", values="count").fillna(0)

# 3. Dodajemy kolumnę informującą o łącznej liczbie utworów w gatunku
explicit_pivot["total_tracks"] = explicit_pivot[True] + explicit_pivot[False]

# Opcjonalne zabezpieczenie: odrzucamy gatunki widma (które mają np. mniej niż 10 utworów w bazie),
# aby pojedyncza piosenka explicit nie dawała sztucznego wyniku 100%
explicit_pivot = explicit_pivot[explicit_pivot["total_tracks"] > 10]

# 4. Obliczamy procentowy udział explicit=True
explicit_pivot["pct_explicit"] = (explicit_pivot[True] / explicit_pivot["total_tracks"] * 100)

# 5. TERAZ wybieramy TOP 20 gatunków, ale sortując po procencie explicit, a nie po liczbie wierszy!
explicit_pivot = explicit_pivot.sort_values("pct_explicit", ascending=False).head(20)

# Rysowanie wykresu
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(explicit_pivot.index, explicit_pivot["pct_explicit"], color="tomato")
ax.set_xlabel("% utworów z treściami wulgarnymi")
ax.set_title("Top 20 najbardziej wulgarnych (explicit) gatunków muzycznych")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("presentation/wykres5_explicit_per_genre.png", dpi=150)
plt.close()
print("Zapisano: wykres5_explicit_per_genre.png z uwzględnieniem Hip-hopu!")
# ==================================================================
# WYKRES 5: Radar chart — profil audio wybranych gatunków
# Ręczna konstrukcja kąt po kącie — np.linspace, wyk. 02/03
# ==================================================================

# Profil audio per gatunek (średnie)
genre_profile = df.groupby("track_genre", observed=True)[audio_features].mean()
genre_profile = genre_profile.reset_index()


def radar_chart(genres, profile_df, features, filename):
    """
    Radar chart (spider plot) dla wybranych gatunków.
    Kąty obliczone przez np.linspace — wyk. 02/03: tablice numpy, wektoryzacja.
    """
    n = len(features)
    # np.linspace — wyk. 02: tworzenie wektorów o równomiernym rozkładzie
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # zamknięcie wykresu (ostatni punkt = pierwszy)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for genre in genres:
        # Filtrowanie wiersza dla danego gatunku — wyk. 04: df.loc[warunek]
        mask = profile_df["track_genre"] == genre
        if mask.sum() == 0:
            print(f"  Ostrzeżenie: gatunek '{genre}' nie znaleziony w danych.")
            continue
        row = profile_df.loc[mask, features].values.flatten().tolist()
        row += row[:1]  # zamknięcie
        ax.plot(angles, row, label=genre, linewidth=2)
        ax.fill(angles, row, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_ylim(0, 1)
    ax.set_title("Profil audio wybranych gatunków", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


selected = ["pop", "metal", "jazz", "classical", "hip-hop"]
# Sprawdzamy które gatunki faktycznie są w danych
available = genre_profile["track_genre"].tolist()
selected = [g for g in selected if g in available]
if not selected:
    # Fallback: wybieramy pierwsze 5 dostępnych gatunków
    selected = available[:5]
print(f"Gatunki do radar chartu: {selected}")

radar_chart(selected, genre_profile, audio_features,
            "presentation/wykres6_radar_audio_profile.png")
print("Zapisano: wykres6_radar_audio_profile.png")
