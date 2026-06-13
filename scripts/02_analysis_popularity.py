import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("presentation", exist_ok=True)

df = pd.read_parquet("data/clean_tracks.parquet")

# ==================================================================
# WYKRES 1: Top 20 artystów wg mediany popularity
# ==================================================================

# Rozbijamy kolumnę artists — str.split + str.strip, wyk. 06
# Analogia z wyk. 06: w.str.strip().str.replace().str.split() na danych
# Kolumna artists zawiera wiele artystów oddzielonych ";"
df_artists = df.copy()
df_artists["artist"] = df_artists["artists"].str.split(";")
df_artists = df_artists.explode("artist")
df_artists["artist"] = df_artists["artist"].str.strip()

# Top 20 artystów — groupby + reset_index + sort_values, wyk. 05
# Analogia: wynik = tips.groupby('day')['total_bill'].mean()
#           wynik = wynik.reset_index()
#           wynik = wynik.sort_values('total_bill', ascending=False)
artist_pop = df_artists.groupby("artist", observed=True)["popularity"].median()
artist_pop = artist_pop.reset_index()
artist_pop = artist_pop.sort_values("popularity", ascending=False)
artist_pop = artist_pop.iloc[:20, :]  # wyk. 04: iloc po pozycji

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(artist_pop["artist"], artist_pop["popularity"], color="steelblue")
ax.set_xlabel("Mediana popularności")
ax.set_title("Top 20 artystów wg mediany popularności")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("presentation/wykres1_top20_artists.png", dpi=150)
plt.close()
print("Zapisano: wykres1_top20_artists.png")

# ==================================================================
# WYKRES 2: Box plot — rozkład popularity per genre (top 15 gatunków)
# wyk. 12: boxplot omówiony na przykładzie iris
# ==================================================================

# value_counts() + head() — wyk. 05: zliczanie kategorii
top_genres = df["track_genre"].value_counts().head(15).index
df_top = df[df["track_genre"].isin(top_genres)]

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df_top, x="track_genre", y="popularity", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_xlabel("Gatunek muzyczny")
ax.set_ylabel("Popularność")
ax.set_title("Rozkład popularności w top 15 gatunkach")
plt.tight_layout()
plt.savefig("presentation/wykres2_popularity_boxplot.png", dpi=150)
plt.close()
print("Zapisano: wykres2_popularity_boxplot.png")

