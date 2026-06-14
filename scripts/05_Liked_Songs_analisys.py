import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tworzenie folderu na wykresy
os.makedirs("presentation", exist_ok=True)

# 1. ŁADOWANIE I CZYSZCZENIE DANYCH
df_aleksander = pd.read_csv("data/Aleksander_Liked_Songs.csv")
df_kacper = pd.read_csv("data/Kacper_Liked_Songs.csv")

df_aleksander["User"] = "Aleksander"
df_kacper["User"] = "Kacper"


def clean_genres(df):
    """Usuwa wiersze z brakującymi gatunkami."""
    df = df.dropna(subset=["Genres"])
    df = df[df["Genres"].str.strip() != ""]
    return df


df_aleksander = clean_genres(df_aleksander)
df_kacper = clean_genres(df_kacper)

df_all = pd.concat([df_aleksander, df_kacper], ignore_index=True)


# ==================================================================
# WYKRES 1: NAJPOPULARNIEJSZE GATUNKI MUZYCZNE (Zastąpiony wykres piosenek)
# ==================================================================
def plot_top_genres_bar(df, user_name, top_n=10):
    # 1. Rozbijamy gatunki, ponieważ piosenka może mieć ich kilka (np. "pop, rock")
    # Zakładamy, że są rozdzielane przecinkami. Czyścimy też białe znaki (strip)
    genres_series = (
        df["Genres"].str.split(",").explode().str.strip()
    )

    # 2. Zliczamy wystąpienia każdego gatunku i bierzemy Top N
    top_genres = genres_series.value_counts().head(top_n)

    if top_genres.empty:
        print(f"Brak danych o gatunkach dla użytkownika {user_name}")
        return

    # Wykres poziomy (barh) jest idealny do długich nazw gatunków
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Kolory od najciemniejszego do najjaśniejszego
    colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(top_genres)))
    
    # Rysujemy słupki (odwracamy kolejność, żeby najwyższy wynik był na górze)
    bars = ax.barh(
        top_genres.index[::-1], 
        top_genres.values[::-1], 
        color=colors[::-1],
        edgecolor="grey"
    )

    # Dodanie wartości liczbowych na końcach słupków
    ax.bar_label(bars, padding=5, fontsize=10)

    ax.set_title(
        f"Top {top_n} najpopularniejszych gatunków muzycznych - {user_name}",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Liczba utworów", fontsize=11)
    
    # Usuwamy zbędne ramki dla lepszego looku
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    filename = f"presentation/wykres_genres_{user_name.lower()}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Zapisano wykres gatunków: {filename}")


# Generowanie wykresów gatunków dla każdego z użytkowników
plot_top_genres_bar(df_aleksander, "Aleksander", top_n=10)
plot_top_genres_bar(df_kacper, "Kacper", top_n=10)


# WYKRES 2: Radar chart — ja vs Kaper

audio_features = [
    "Danceability",
    "Energy",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
    "Valence",
]

user_profile = df_all.groupby("User", observed=True)[audio_features].mean()
user_profile = user_profile.reset_index()


def radar_chart_users(users, profile_df, features, filename):
    n = len(features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {"Aleksander": "steelblue", "Kacper": "darkorange"}

    for user in users:
        mask = profile_df["User"] == user
        if mask.sum() == 0:
            continue

        row = profile_df.loc[mask, features].values.flatten().tolist()
        row += row[:1]

        ax.plot(angles, row, label=user, linewidth=2, color=colors.get(user))
        ax.fill(angles, row, alpha=0.15, color=colors.get(user))

    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_ylim(0, 1)
    ax.set_title("Porównanie profilu audio: Aleksander vs Kacper", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Zapisano: {filename}")


radar_chart_users(
    ["Aleksander", "Kacper"],
    user_profile,
    audio_features,
    "presentation/wykres_radar_audio_profile_users.png",
)